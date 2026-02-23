"""
portfolio_engine.py
-------------------
Main event-driven simulation loop.

D1 strategies:
  Each business day: execute pending -> trailing stops -> signals -> MTM

H4 strategies:
  Each H4 bar: execute pending -> trailing stops -> check D1 trend + H4 signal -> MTM (daily)

Modes (H4 strategies only):
  Mode A — Ideal Robot:
      Take every valid signal. Enter at next H4 bar open. No constraints.

  Mode B — Human Executable:
      Sleep window: only enter during 09:00–23:00 local time.
      Too-late filter: cancel if actual entry price deviates > 1×ATR(signal) from ref entry.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from data.loader import load_all_data
from data.mt5_data_loader import INSTRUMENTS as _INSTRUMENTS, BUCKET_MAP
from engine.signal_engine import SignalEngine, SignalResult, precompute_indicators
from engine.risk_engine import RiskEngine, Trade
from strategies.base import Strategy


INSTRUMENTS = _INSTRUMENTS

# ── Sleep-window / too-late utilities (importable for unit tests) ─────────────

def is_in_sleep_window(ts: pd.Timestamp) -> bool:
    """Return True if timestamp falls within trader active hours [09:00, 23:00)."""
    return 9 <= ts.hour < 23


def check_too_late(actual_price: float, ref_price: float, atr_signal: float,
                   multiplier: float = 1.0) -> bool:
    """Return True if the trade should be cancelled (price drifted too far).

    Uses atr_signal (ATR at signal time, not entry time) per spec.
    """
    return abs(actual_price - ref_price) > multiplier * atr_signal


# ── Mode B pending signal container ──────────────────────────────────────────

@dataclass
class _HumanPending:
    """Wraps a SignalResult with additional Mode B execution metadata."""
    sig:            SignalResult
    atr_signal:     float                    # ATR frozen at signal bar
    signal_ts:      pd.Timestamp             # H4 bar when signal fired
    ref_price:      Optional[float] = None   # Open of first bar after signal (Mode A ref)
    ref_ts:         Optional[pd.Timestamp] = None
    bars_since_sig: int = 0                  # H4 bars processed after signal bar


# ── Daily equity snapshot ─────────────────────────────────────────────────────

@dataclass
class DailySnapshot:
    date:           pd.Timestamp
    equity:         float          # settled cash equity
    total_value:    float          # equity + open MTM
    open_trades:    int
    total_r_open:   float
    new_trades:     int
    closed_trades:  int
    signals_seen:   int
    signals_rejected: int


class PortfolioEngine:
    def __init__(self,
                 strategy: Strategy,
                 start: str = "2008-01-01",
                 end:   str = "2024-12-31",
                 initial_equity: float = 100_000.0,
                 instruments: list | None = None,
                 mode: str = "A",
                 cap_per_bucket: float = 1.0,
                 cap_total: float | None = None):
        """
        mode: "A" = Ideal Robot (default), "B" = Human Executable.
        Mode B applies sleep-window and too-late ATR filter (H4 strategies only).
        cap_per_bucket: max R open per bucket (default 1.0 = one trade per bucket).
        cap_total: max total R open across all buckets (default = n_active_buckets × cap_per_bucket).
        """
        self.strategy       = strategy
        self.start          = pd.Timestamp(start)
        self.end            = pd.Timestamp(end)
        self.initial_equity = initial_equity
        self.mode           = mode.upper()
        self._cap_per_bucket = cap_per_bucket

        cfg = strategy.config()
        universe = instruments if instruments is not None else INSTRUMENTS

        print("Loading market data...")
        warmup_start = self.start - pd.DateOffset(months=6)
        self.raw_data = load_all_data(str(warmup_start.date()), end)

        self.active_instruments = [n for n in universe if n in self.raw_data]

        print("Pre-computing indicators...")
        self.indicators: Dict[str, Dict[str, pd.DataFrame]] = {}
        for name in self.active_instruments:
            h4 = self.raw_data[name].get("H4") if strategy.uses_h4 else None
            self.indicators[name] = precompute_indicators(
                strategy,
                self.raw_data[name]["D1"],
                h4=h4,
            )

        # Count distinct active buckets to set default total cap
        from engine.risk_engine import BUCKET_MAP as _BM
        active_buckets = {_BM[n] for n in self.active_instruments if n in _BM}
        _cap_total = cap_total if cap_total is not None else len(active_buckets) * cap_per_bucket

        self.signal_engine = SignalEngine(strategy, self.indicators,
                                          self.active_instruments)
        self.risk_engine   = RiskEngine(initial_equity, config=cfg,
                                        cap_per_bucket=cap_per_bucket,
                                        cap_total=_cap_total)

        self.equity_curve:      List[DailySnapshot] = []
        self.all_trades:        List[Trade]          = []
        self.pending_signals:   Dict[str, SignalResult] = {}
        self.pending_exits:     set[str]             = set()

        # Mode B statistics
        self.cancelled_sleep:    int = 0   # cancelled: no eligible bar within 24h
        self.cancelled_too_late: int = 0   # cancelled: price drifted > 1×ATR
        self.delay_distribution: Dict[int, int] = {}  # {n_bars_delayed: count}
        self.cancelled_samples:  List[dict] = []       # up to 5 cancelled trade samples

    # ── Main entry point ─────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """Execute the full backtest. Returns equity curve as DataFrame."""
        if self.strategy.uses_h4:
            return self._run_h4()
        return self._run_d1()

    # ── D1 simulation loop ───────────────────────────────────────────────────
    def _run_d1(self) -> pd.DataFrame:
        cfg = self.strategy.config()
        print(f"\nRunning backtest {self.start.date()} -> {self.end.date()}...")

        trading_days = pd.bdate_range(self.start, self.end)

        for date in trading_days:
            date = pd.Timestamp(date)

            d1_bars: Dict[str, pd.Series] = {}
            for name in self.active_instruments:
                d1_df = self.indicators[name]["D1"]
                if date in d1_df.index:
                    d1_bars[name] = d1_df.loc[date]

            if not d1_bars:
                continue

            # Step 1: Execute pending signals at D1 open
            new_trades_today = 0
            rejected_today   = 0

            for name in list(self.pending_signals):
                sig = self.pending_signals.pop(name)
                bar = d1_bars.get(name)
                if bar is None:
                    rejected_today += 1
                    continue
                sig.entry_price = float(bar["open"])
                sig.date = date
                trade = self.risk_engine.open_trade(sig)
                if trade is not None:
                    new_trades_today += 1
                else:
                    rejected_today += 1

            # Step 1b: Execute pending channel exits at D1 open
            for inst_name in list(self.pending_exits):
                self.pending_exits.discard(inst_name)
                bar = d1_bars.get(inst_name)
                if bar is None:
                    continue
                for trade in list(self.risk_engine.state.open_trades):
                    if trade.instrument == inst_name:
                        self.risk_engine._close_trade(
                            trade, date, float(bar["open"]), "channel_exit")
                        self.all_trades.append(trade)
                        break

            # Step 2: Trailing stops
            closed_today = self.risk_engine.update_trailing_stops_d1(date, d1_bars)
            self.all_trades.extend(closed_today)

            # Step 3: Generate signals
            signals = self.signal_engine.get_signals(date)
            signals_seen = 0

            for name, sig in signals.items():
                if sig is None:
                    continue
                signals_seen += 1

                if cfg.use_reversal_exit:
                    reversed_trade = self.risk_engine.close_on_reversal(
                        date, name, sig.direction, d1_bars
                    )
                    if reversed_trade is not None:
                        self.all_trades.append(reversed_trade)

                if name not in self.pending_signals:
                    self.pending_signals[name] = sig

            # Step 3b: Check channel exit signals
            for trade in list(self.risk_engine.state.open_trades):
                bar = d1_bars.get(trade.instrument)
                if bar is None:
                    continue
                if self.strategy.get_exit_signal(date, bar, trade.direction):
                    self.pending_exits.add(trade.instrument)

            # Step 4: MTM + snapshot
            total_value = self.risk_engine.mark_to_market(d1_bars)
            self.equity_curve.append(DailySnapshot(
                date=date, equity=self.risk_engine.state.equity,
                total_value=total_value,
                open_trades=len(self.risk_engine.state.open_trades),
                total_r_open=self.risk_engine.state.open_r_exposure(),
                new_trades=new_trades_today, closed_trades=len(closed_today),
                signals_seen=signals_seen, signals_rejected=rejected_today,
            ))

        self._close_remaining(trading_days[-1])
        print(f"Backtest complete. {len(self.all_trades)} total trades processed.")
        return self._equity_curve_df()

    # ── H4 simulation loop ───────────────────────────────────────────────────
    def _run_h4(self) -> pd.DataFrame:
        cfg    = self.strategy.config()
        mode_b = (self.mode == "B")

        print(f"\nRunning H4 backtest [Mode {self.mode}] "
              f"{self.start.date()} -> {self.end.date()}...")

        # Mode B uses its own pending dict so Mode A state is untouched
        pending_b: Dict[str, _HumanPending] = {}

        # Build sorted timeline of all H4 timestamps in trading period
        all_h4_times = set()
        for name in self.active_instruments:
            h4_df = self.indicators[name].get("H4")
            if h4_df is not None:
                mask = (h4_df.index >= self.start) & (h4_df.index <= self.end + pd.Timedelta(days=1))
                all_h4_times.update(h4_df.index[mask])
        h4_timeline = sorted(all_h4_times)

        if not h4_timeline:
            print("No H4 bars in trading period.")
            return self._equity_curve_df()

        # Pre-index D1 dates per instrument for fast trend lookup
        d1_indices = {}
        for name in self.active_instruments:
            d1_indices[name] = self.indicators[name]["D1"].index

        prev_snap_date = None
        day_new_trades = day_closed = day_signals = day_rejected = 0

        for h4_ts in h4_timeline:
            current_date = h4_ts.normalize()

            # Day boundary → snapshot previous day
            if prev_snap_date is not None and current_date != prev_snap_date:
                self._snapshot_day(prev_snap_date, day_new_trades, day_closed,
                                   day_signals, day_rejected)
                day_new_trades = day_closed = day_signals = day_rejected = 0

            prev_snap_date = current_date

            # Build H4 bar lookup for this timestamp
            h4_bars: Dict[str, pd.Series] = {}
            for name in self.active_instruments:
                h4_df = self.indicators[name].get("H4")
                if h4_df is not None and h4_ts in h4_df.index:
                    h4_bars[name] = h4_df.loc[h4_ts]

            if not h4_bars:
                continue

            # ── Step 1: Execute pending signals ──────────────────────────────

            if not mode_b:
                # Mode A: enter at next H4 bar open, no constraints
                for name in list(self.pending_signals):
                    sig = self.pending_signals.pop(name)
                    bar = h4_bars.get(name)
                    if bar is None:
                        day_rejected += 1
                        continue
                    sig.entry_price = float(bar["open"])
                    sig.date = h4_ts
                    trade = self.risk_engine.open_trade(sig)
                    if trade is not None:
                        day_new_trades += 1
                    else:
                        day_rejected += 1

            else:
                # Mode B: sleep window + too-late ATR filter
                for name in list(pending_b):
                    hp  = pending_b[name]
                    bar = h4_bars.get(name)
                    if bar is None:
                        # No data for this instrument at this bar — keep waiting
                        continue

                    hp.bars_since_sig += 1
                    bar_open = float(bar["open"])

                    # First bar after signal: record Mode A reference entry price
                    if hp.ref_price is None:
                        hp.ref_price = bar_open
                        hp.ref_ts    = h4_ts

                    in_window = is_in_sleep_window(h4_ts)

                    if in_window:
                        # First eligible bar found — apply too-late filter once
                        distance = abs(bar_open - hp.ref_price)
                        if check_too_late(bar_open, hp.ref_price, hp.atr_signal):
                            # Cancel: price drifted too far from ref entry
                            self.cancelled_too_late += 1
                            if len(self.cancelled_samples) < 5:
                                self.cancelled_samples.append({
                                    "reason":             "too_late",
                                    "instrument":         hp.sig.instrument,
                                    "signal_ts":          hp.signal_ts,
                                    "ref_entry_price":    hp.ref_price,
                                    "actual_entry_price": bar_open,
                                    "atr_signal":         hp.atr_signal,
                                    "distance":           distance,
                                })
                            del pending_b[name]
                        else:
                            # Enter trade at this bar's open
                            hp.sig.entry_price = bar_open
                            hp.sig.date        = h4_ts
                            trade = self.risk_engine.open_trade(hp.sig)
                            # Delay = 0 means same-speed as Mode A (entered on ref bar)
                            delay = hp.bars_since_sig - 1
                            self.delay_distribution[delay] = (
                                self.delay_distribution.get(delay, 0) + 1
                            )
                            if trade is not None:
                                day_new_trades += 1
                            else:
                                day_rejected += 1
                            del pending_b[name]

                    else:
                        # Outside sleep window — check 24h timeout
                        if (h4_ts - hp.signal_ts) > pd.Timedelta(hours=24):
                            self.cancelled_sleep += 1
                            if len(self.cancelled_samples) < 5:
                                self.cancelled_samples.append({
                                    "reason":             "sleep_window",
                                    "instrument":         hp.sig.instrument,
                                    "signal_ts":          hp.signal_ts,
                                    "ref_entry_price":    hp.ref_price,
                                    "actual_entry_price": bar_open,
                                    "atr_signal":         hp.atr_signal,
                                    "distance":           (
                                        abs(bar_open - hp.ref_price)
                                        if hp.ref_price is not None else None
                                    ),
                                })
                            del pending_b[name]

            # ── Step 2: Update trailing stops ────────────────────────────────
            closed = self.risk_engine.update_trailing_stops_d1(h4_ts, h4_bars)
            self.all_trades.extend(closed)
            day_closed += len(closed)

            # ── Step 3: Check H4 signals with D1 trend filter ────────────────
            for name in self.active_instruments:
                h4_bar = h4_bars.get(name)
                if h4_bar is None:
                    continue

                # Skip if open trade or already pending
                if self.risk_engine.state.instrument_has_open_trade(name):
                    continue
                if not mode_b and name in self.pending_signals:
                    continue
                if mode_b and name in pending_b:
                    continue

                # D1 trend from most recent completed D1 bar
                d1_df     = self.indicators[name]["D1"]
                d1_before = d1_indices[name][d1_indices[name] < current_date]
                if len(d1_before) == 0:
                    continue
                d1_row    = d1_df.loc[d1_before[-1]]

                d1_trend = self.strategy.get_d1_trend(d1_row)
                if d1_trend is None:
                    continue

                direction = self.strategy.get_h4_signal(h4_bar, d1_trend)
                if direction is None:
                    continue

                day_signals += 1
                atr_val = float(h4_bar["atr"])

                sig = SignalResult(
                    instrument  = name,
                    date        = h4_ts,
                    direction   = direction,
                    atr         = atr_val,
                    entry_price = float(h4_bar["close"]),
                    d1_close    = float(d1_row["close"]),
                )

                if not mode_b:
                    self.pending_signals[name] = sig
                else:
                    pending_b[name] = _HumanPending(
                        sig        = sig,
                        atr_signal = atr_val,   # frozen at signal time
                        signal_ts  = h4_ts,
                    )

        # Final day snapshot
        if prev_snap_date is not None:
            self._snapshot_day(prev_snap_date, day_new_trades, day_closed,
                               day_signals, day_rejected)

        # Close remaining open trades at last available price
        last_ts = h4_timeline[-1]
        h4_bars_last: Dict[str, pd.Series] = {}
        for name in self.active_instruments:
            h4_df = self.indicators[name].get("H4")
            if h4_df is not None and last_ts in h4_df.index:
                h4_bars_last[name] = h4_df.loc[last_ts]
        for trade in list(self.risk_engine.state.open_trades):
            bar = h4_bars_last.get(trade.instrument)
            if bar is not None:
                self.risk_engine._close_trade(
                    trade, last_ts, float(bar["close"]), "end_of_backtest")
        self.all_trades.extend(self.risk_engine.state.closed_trades)

        print(f"Backtest complete. {len(self.all_trades)} total trades processed.")
        return self._equity_curve_df()

    def _snapshot_day(self, date, new_trades, closed, signals, rejected):
        """Create a daily equity snapshot using last known bar prices."""
        mtm_bars: Dict[str, pd.Series] = {}
        for name in self.active_instruments:
            d1_df = self.indicators[name]["D1"]
            d1_on_date = d1_df.index[d1_df.index.normalize() == date]
            if len(d1_on_date) > 0:
                mtm_bars[name] = d1_df.loc[d1_on_date[-1]]
            else:
                h4_df = self.indicators[name].get("H4")
                if h4_df is not None:
                    h4_on_date = h4_df.index[h4_df.index.normalize() == date]
                    if len(h4_on_date) > 0:
                        mtm_bars[name] = h4_df.loc[h4_on_date[-1]]

        total_value = self.risk_engine.mark_to_market(mtm_bars)
        self.equity_curve.append(DailySnapshot(
            date=date, equity=self.risk_engine.state.equity,
            total_value=total_value,
            open_trades=len(self.risk_engine.state.open_trades),
            total_r_open=self.risk_engine.state.open_r_exposure(),
            new_trades=new_trades, closed_trades=closed,
            signals_seen=signals, signals_rejected=rejected,
        ))

    # ── Mode B statistics ─────────────────────────────────────────────────────
    def get_mode_b_stats(self) -> dict:
        """Return Mode B cancellation and delay statistics."""
        return {
            "cancelled_sleep_window": self.cancelled_sleep,
            "cancelled_too_late":     self.cancelled_too_late,
            "delay_distribution":     dict(sorted(self.delay_distribution.items())),
            "cancelled_samples":      self.cancelled_samples,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _close_remaining(self, last_date):
        d1_bars: Dict[str, pd.Series] = {}
        for name in self.active_instruments:
            d1_df = self.indicators[name]["D1"]
            if last_date in d1_df.index:
                d1_bars[name] = d1_df.loc[last_date]
        for trade in list(self.risk_engine.state.open_trades):
            bar = d1_bars.get(trade.instrument)
            if bar is not None:
                self.risk_engine._close_trade(
                    trade, last_date, float(bar["close"]), "end_of_backtest")
        self.all_trades.extend(self.risk_engine.state.closed_trades)

    def _equity_curve_df(self) -> pd.DataFrame:
        rows = [
            {
                "date":             s.date,
                "equity":           s.equity,
                "total_value":      s.total_value,
                "open_trades":      s.open_trades,
                "total_r_open":     s.total_r_open,
                "new_trades":       s.new_trades,
                "closed_trades":    s.closed_trades,
                "signals_seen":     s.signals_seen,
                "signals_rejected": s.signals_rejected,
            }
            for s in self.equity_curve
        ]
        return pd.DataFrame(rows).set_index("date")

    def get_trades_df(self) -> pd.DataFrame:
        rows = []
        for t in self.all_trades:
            rows.append({
                "trade_id":     t.trade_id,
                "instrument":   t.instrument,
                "bucket":       t.bucket,
                "direction":    t.direction,
                "entry_date":   t.entry_date,
                "exit_date":    t.exit_date,
                "entry_price":  t.entry_price,
                "exit_price":   t.exit_price,
                "units":        t.units,
                "pnl":          t.pnl,
                "r_multiple":   t.r_multiple,
                "exit_reason":  t.exit_reason,
                "entry_equity": t.entry_equity,
                "hold_days":    (t.exit_date - t.entry_date).days if t.exit_date else None,
            })
        return pd.DataFrame(rows)
