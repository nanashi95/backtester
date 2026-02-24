"""
ensemble_engine.py
------------------
Multi-sleeve portfolio engine for the 3-horizon Donchian ensemble.

Architecture:
  - One shared equity account across all sleeves (A, B, C)
  - Three independent trade books — one position per instrument per sleeve
  - Global portfolio cap enforced across all open trades combined
  - No per-bucket caps — pure global R cap only
  - Signal priority when cap-constrained: Sleeve A first, then B, then C

Daily simulation order (D1):
  1. Execute pending signals at today's open (priority A → B → C)
  2. Update ATR trailing stops for all sleeves
  3. Generate new signals for all sleeves, queue for tomorrow
  4. Mark-to-market snapshot
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data.loader import load_all_data, get_instrument_bucket
from engine.signal_engine import SignalEngine, SignalResult, precompute_indicators
from engine.risk_engine import Trade
from strategies.base import Strategy, StrategyConfig


RISK_PER_TRADE_DEFAULT = 0.007   # 0.7% per trade
GLOBAL_CAP_DEFAULT     = 6.0    # max total R open across all sleeves
SLEEVES                = ["A", "B", "C"]


# ── Shared-equity risk manager ────────────────────────────────────────────────

class EnsembleRiskManager:
    """
    Single equity account shared across all sleeves.
    Enforces global portfolio cap; no per-bucket caps.
    """

    def __init__(self,
                 initial_equity:  float,
                 sleeve_configs:  Dict[str, StrategyConfig],
                 risk_per_trade:  float = RISK_PER_TRADE_DEFAULT,
                 global_cap:      float = GLOBAL_CAP_DEFAULT,
                 sleeves:         Optional[List[str]] = None,
                 sector_cap_pct:  float = 0.0,
                 use_vol_scaling: bool  = False):
        self.equity         = initial_equity
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.global_cap     = global_cap
        self.sleeve_configs = sleeve_configs
        self.sleeves        = sleeves if sleeves is not None else SLEEVES
        self.sector_cap_pct  = sector_cap_pct
        self.use_vol_scaling = use_vol_scaling

        self.open_trades:   Dict[str, List[Trade]] = {s: [] for s in self.sleeves}
        self.closed_trades: List[Trade] = []
        self.sector_cap_rejections: Dict[str, int] = {}
        self._counter = 0

    # ── Aggregate helpers ─────────────────────────────────────────────────────

    def all_open(self) -> List[Trade]:
        return [t for s in self.sleeves for t in self.open_trades[s]]

    def open_r(self) -> float:
        return sum(t.r_risked for t in self.all_open())

    def sleeve_has_open(self, sleeve: str, instrument: str) -> bool:
        return any(t.instrument == instrument for t in self.open_trades[sleeve])

    def sector_r(self, bucket: str) -> float:
        return sum(t.r_risked for t in self.all_open() if t.bucket == bucket)

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def can_open(self, sleeve: str, instrument: str) -> Tuple[bool, str]:
        if self.sleeve_has_open(sleeve, instrument):
            return False, "sleeve_dup"
        if self.open_r() + 1.0 > self.global_cap + 1e-9:
            return False, "global_cap"
        if self.sector_cap_pct > 0.0:
            bucket = get_instrument_bucket(instrument)
            cap    = self.global_cap * self.sector_cap_pct
            if self.sector_r(bucket) + 1.0 > cap + 1e-9:
                return False, f"sector_cap_{bucket}"
        return True, ""

    def open_trade(self, sleeve: str, sig: SignalResult) -> Optional[Trade]:
        ok, reason = self.can_open(sleeve, sig.instrument)
        if not ok:
            if reason.startswith("sector_cap_"):
                bucket = reason[len("sector_cap_"):]
                self.sector_cap_rejections[bucket] = (
                    self.sector_cap_rejections.get(bucket, 0) + 1
                )
            return None

        cfg       = self.sleeve_configs[sleeve]
        stop_dist = cfg.atr_initial_stop * sig.atr
        if stop_dist <= 0:
            return None

        # Vol-based risk scaling (only when use_vol_scaling=True)
        # Low-vol  (pct < 0.33): expand size  → 1.2×
        # Mid-vol  (0.33–0.66):  base size    → 1.0×
        # High-vol (pct > 0.66): compress     → 0.6×
        if self.use_vol_scaling:
            pct = sig.atr_percentile
            if pct == pct:  # not NaN
                if pct < 0.33:
                    multiplier = 1.2
                elif pct > 0.66:
                    multiplier = 0.6
                else:
                    multiplier = 1.0
            else:
                multiplier = 1.0
        else:
            multiplier = 1.0

        dollar_risk  = self.equity * self.risk_per_trade * multiplier
        units        = dollar_risk / stop_dist
        initial_stop = sig.entry_price - sig.direction * stop_dist

        self._counter += 1
        trade = Trade(
            trade_id        = self._counter,
            instrument      = sig.instrument,
            bucket          = get_instrument_bucket(sig.instrument),
            direction       = sig.direction,
            entry_date      = sig.date,
            entry_price     = sig.entry_price,
            units           = units,
            initial_stop    = initial_stop,
            trailing_stop   = initial_stop,
            highest_fav     = sig.entry_price,
            r_risked        = 1.0,
            entry_equity    = self.equity,
            sleeve          = sleeve,
            risk_multiplier = multiplier,
        )
        self.open_trades[sleeve].append(trade)
        return trade

    def update_trailing_stops(self,
                               date:    pd.Timestamp,
                               d1_bars: Dict[str, pd.Series],
                               sleeve:  str) -> List[Trade]:
        cfg    = self.sleeve_configs[sleeve]
        closed: List[Trade] = []

        for trade in list(self.open_trades[sleeve]):
            bar = d1_bars.get(trade.instrument)
            if bar is None:
                continue
            atr_val = bar.get("atr")
            if atr_val is None or pd.isna(atr_val):
                continue

            trail_dist = cfg.atr_trailing_stop * atr_val
            high = float(bar["high"])
            low  = float(bar["low"])

            if trade.direction == 1:
                if high > trade.highest_fav:
                    trade.highest_fav  = high
                    new_trail = trade.highest_fav - trail_dist
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)
                if date != trade.entry_date and low <= trade.trailing_stop:
                    self._close(trade, date, trade.trailing_stop, "trailing_stop")
                    closed.append(trade)
            else:  # short
                if low < trade.highest_fav:
                    trade.highest_fav  = low
                    new_trail = trade.highest_fav + trail_dist
                    trade.trailing_stop = min(trade.trailing_stop, new_trail)
                if date != trade.entry_date and high >= trade.trailing_stop:
                    self._close(trade, date, trade.trailing_stop, "trailing_stop")
                    closed.append(trade)

        for t in closed:
            self.open_trades[sleeve].remove(t)
        return closed

    def _close(self, trade: Trade, date: pd.Timestamp,
               price: float, reason: str) -> None:
        # Original units PnL
        pnl_orig = trade.pnl_from_price(price)

        # Pyramid units PnL (different entry price)
        pnl_pyr  = 0.0
        risk_pyr = 0.0
        if trade.pyramid_added and trade.pyramid_entry_price is not None:
            pnl_pyr  = ((price - trade.pyramid_entry_price)
                        * trade.direction * trade.pyramid_units)
            risk_pyr = trade.pyramid_dollar_risk  # stored at add time

        total_pnl  = pnl_orig + pnl_pyr
        risk_orig  = trade.entry_equity * self.risk_per_trade * trade.risk_multiplier
        total_risk = risk_orig + risk_pyr
        r_multiple = total_pnl / total_risk if total_risk > 0 else 0.0

        trade.exit_date   = date
        trade.exit_price  = price
        trade.pnl         = total_pnl
        trade.r_multiple  = r_multiple
        trade.exit_reason = reason

        self.equity += total_pnl
        self.closed_trades.append(trade)

    def mark_to_market(self, d1_bars: Dict[str, pd.Series]) -> float:
        mtm = self.equity
        for trade in self.all_open():
            bar = d1_bars.get(trade.instrument)
            if bar is not None:
                close = float(bar["close"])
                mtm += trade.pnl_from_price(close)
                # Add pyramid unit unrealised PnL
                if trade.pyramid_added and trade.pyramid_entry_price is not None:
                    mtm += ((close - trade.pyramid_entry_price)
                            * trade.direction * trade.pyramid_units)
        return mtm

    def close_remaining(self, date: pd.Timestamp,
                        d1_bars: Dict[str, pd.Series]) -> None:
        for sleeve in self.sleeves:
            for trade in list(self.open_trades[sleeve]):
                bar   = d1_bars.get(trade.instrument)
                price = float(bar["close"]) if bar is not None else trade.trailing_stop
                self._close(trade, date, price, "end_of_backtest")
            self.open_trades[sleeve].clear()


# ── Daily snapshot ─────────────────────────────────────────────────────────────

@dataclass
class DailySnapshot:
    date:         pd.Timestamp
    equity:       float
    total_value:  float
    open_trades:  int
    total_r_open: float
    new_trades:   int
    closed_trades: int


# ── Ensemble portfolio engine ─────────────────────────────────────────────────

class EnsemblePortfolioEngine:
    """
    D1 event loop running 3 Donchian sleeves simultaneously on a shared account.
    Data loaded once; indicators precomputed per sleeve.
    """

    def __init__(self,
                 strategies:         Dict[str, Strategy],
                 start:              str,
                 end:                str,
                 initial_equity:     float = 100_000.0,
                 instruments:        Optional[List[str]] = None,
                 risk_per_trade:     float = RISK_PER_TRADE_DEFAULT,
                 global_cap:         float = GLOBAL_CAP_DEFAULT,
                 sleeve_instruments:    Optional[Dict[str, List[str]]] = None,
                 use_pyramiding:        bool  = False,
                 pyramid_min_days:      int   = 20,
                 pyramid_risk_fraction: float = 1.0,
                 sector_cap_pct:        float = 0.0,
                 use_vol_scaling:       bool  = False,
                 dd_pause_pct:          float = 0.0):
        """
        sleeve_instruments: per-sleeve instrument filter.
          e.g. {"FX": ["EURUSD","GBPUSD"], "B": ["Gold","US100"]}
          Sleeves not listed trade the full universe.
          Default None = all sleeves trade all instruments.
        """
        self.start          = pd.Timestamp(start)
        self.end            = pd.Timestamp(end)
        self.initial_equity = initial_equity

        active_sleeves = list(strategies.keys())
        sleeve_configs = {s: strategies[s].config() for s in active_sleeves}

        # Load data once for all sleeves
        warmup_start = self.start - pd.DateOffset(months=6)
        raw_data     = load_all_data(str(warmup_start.date()), end)

        universe = instruments if instruments is not None else list(raw_data.keys())
        self.active_instruments = [n for n in universe if n in raw_data]

        # Precompute indicators per sleeve — each sleeve gets its own D1 DataFrame
        # (same OHLC, different indicator columns)
        self._d1_indicators: Dict[str, Dict[str, pd.DataFrame]] = {}
        for sleeve, strategy in strategies.items():
            self._d1_indicators[sleeve] = {}
            for name in self.active_instruments:
                d1 = precompute_indicators(strategy, raw_data[name]["D1"])["D1"]
                self._d1_indicators[sleeve][name] = d1

        # Per-sleeve instrument scope (subset of active_instruments)
        sleeve_inst_map: Dict[str, List[str]] = {}
        for sleeve in active_sleeves:
            if sleeve_instruments and sleeve in sleeve_instruments:
                sleeve_inst_map[sleeve] = [
                    n for n in sleeve_instruments[sleeve]
                    if n in self.active_instruments
                ]
            else:
                sleeve_inst_map[sleeve] = self.active_instruments

        # Signal engines — one per sleeve, scoped to sleeve's instrument subset
        self.signal_engines: Dict[str, SignalEngine] = {}
        for sleeve, strategy in strategies.items():
            si   = sleeve_inst_map[sleeve]
            inds = {name: {"D1": self._d1_indicators[sleeve][name]}
                    for name in si if name in self._d1_indicators[sleeve]}
            self.signal_engines[sleeve] = SignalEngine(strategy, inds, si)

        self.risk    = EnsembleRiskManager(initial_equity, sleeve_configs,
                                           risk_per_trade, global_cap,
                                           sleeves=active_sleeves,
                                           sector_cap_pct=sector_cap_pct,
                                           use_vol_scaling=use_vol_scaling)

        self.use_pyramiding        = use_pyramiding
        self.pyramid_min_days      = pyramid_min_days
        self.pyramid_risk_fraction = pyramid_risk_fraction
        self.dd_pause_pct          = dd_pause_pct

        # Pending signals: {sleeve: {instrument: SignalResult}}
        self.pending: Dict[str, Dict[str, SignalResult]] = {s: {} for s in active_sleeves}

        self._snapshots: List[DailySnapshot] = []

    # ── Pyramiding ───────────────────────────────────────────────────────────

    def _try_pyramid(self, sleeve: str, trade: Trade,
                     date: pd.Timestamp,
                     d1_bars: Dict[str, pd.Series]) -> bool:
        """
        Attempt to add one pyramid unit to an open trade.
        Conditions:
          1. No prior pyramid on this trade
          2. Trade has been open >= pyramid_min_days trading days
          3. Unrealised PnL (original units) >= +2R
          4. Donchian trend still intact (price not through channel)
          5. Portfolio cap can absorb pyramid_risk_fraction R
          6. Stop distance > 0
        Returns True if pyramid was applied.
        """
        if trade.pyramid_added:
            return False

        bar = d1_bars.get(trade.instrument)
        if bar is None:
            return False

        # 1. Persistence: must be held >= pyramid_min_days trading days
        days_held = int(np.busday_count(trade.entry_date.date(), date.date()))
        if days_held < self.pyramid_min_days:
            return False

        close = float(bar["close"])

        # 2. Unrealised PnL check (original units only)
        pnl_orig = (close - trade.entry_price) * trade.direction * trade.units
        one_r    = trade.entry_equity * self.risk.risk_per_trade * trade.risk_multiplier
        if pnl_orig < 2.0 * one_r:
            return False

        # 3. Trend-intact check via sleeve's signal engine
        sig_eng = self.signal_engines.get(sleeve)
        if sig_eng is None:
            return False
        if not sig_eng.is_trend_intact(trade.instrument, date, trade.direction):
            return False

        # 4. Portfolio cap: adding pyramid_risk_fraction R
        if self.risk.open_r() + self.pyramid_risk_fraction > self.risk.global_cap + 1e-9:
            return False

        # 5. Stop distance must be positive
        stop_dist = abs(close - trade.trailing_stop)
        if stop_dist <= 0:
            return False

        # Apply pyramid: sized to pyramid_risk_fraction × base risk
        pyr_dollar_risk = (self.risk.equity * self.risk.risk_per_trade
                           * trade.risk_multiplier * self.pyramid_risk_fraction)
        pyr_units = pyr_dollar_risk / stop_dist

        trade.pyramid_added        = True
        trade.pyramid_units        = pyr_units
        trade.pyramid_entry_price  = close
        trade.pyramid_entry_equity = self.risk.equity
        trade.pyramid_dollar_risk  = pyr_dollar_risk
        trade.r_risked             = 1.0 + self.pyramid_risk_fraction

        return True

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        trading_days  = pd.bdate_range(self.start, self.end)
        _ref_sleeve   = self.risk.sleeves[0]   # OHLC identical across sleeves
        n_sl          = len(self.risk.sleeves)
        print(f"\nRunning ensemble {self.start.date()} -> {self.end.date()} "
              f"({len(self.active_instruments)} instruments, {n_sl} sleeves)...")

        hwm               = self.initial_equity
        dd_paused         = False   # gate state carried from previous EOD
        self._dd_pause_days = 0

        for date in trading_days:
            date = pd.Timestamp(date)

            # Build D1 bar lookup — OHLC identical across sleeves
            d1_bars: Dict[str, pd.Series] = {}
            for name in self.active_instruments:
                d1_df = self._d1_indicators[_ref_sleeve][name]
                if date in d1_df.index:
                    d1_bars[name] = d1_df.loc[date]

            if not d1_bars:
                continue

            new_today    = 0
            closed_today = 0

            # ── 1. Execute pending signals at today's open (priority order) ───
            if not dd_paused:
                for sleeve in self.risk.sleeves:
                    for name in list(self.pending[sleeve]):
                        sig = self.pending[sleeve].pop(name)
                        bar = d1_bars.get(name)
                        if bar is None:
                            continue
                        sig.entry_price = float(bar["open"])
                        sig.date        = date
                        trade = self.risk.open_trade(sleeve, sig)
                        if trade is not None:
                            new_today += 1
            else:
                # Flush stale pending signals — don't carry them into recovery
                for sleeve in self.risk.sleeves:
                    self.pending[sleeve].clear()
                self._dd_pause_days += 1

            # ── 2. Update trailing stops for all sleeves ──────────────────────
            for sleeve in self.risk.sleeves:
                closed = self.risk.update_trailing_stops(date, d1_bars, sleeve)
                closed_today += len(closed)

            # ── 2.5 Check pyramid opportunities (EOD close) ───────────────────
            if self.use_pyramiding:
                for sleeve in self.risk.sleeves:
                    for trade in list(self.risk.open_trades[sleeve]):
                        self._try_pyramid(sleeve, trade, date, d1_bars)

            # ── 2.7 Strategy-level exit signals (EOD close) ───────────────────
            for sleeve in self.risk.sleeves:
                sig_eng = self.signal_engines.get(sleeve)
                if sig_eng is None:
                    continue
                exited = []
                for trade in list(self.risk.open_trades[sleeve]):
                    bar = d1_bars.get(trade.instrument)
                    if bar is None:
                        continue
                    if sig_eng.check_exit_signal(trade.instrument, date,
                                                 trade.direction):
                        exit_price = float(bar["close"])
                        self.risk._close(trade, date, exit_price, "exit_signal")
                        exited.append(trade)
                        closed_today += 1
                for t in exited:
                    self.risk.open_trades[sleeve].remove(t)

            # ── 3. Generate signals → queue for tomorrow ──────────────────────
            if not dd_paused:
                for sleeve in self.risk.sleeves:
                    signals = self.signal_engines[sleeve].get_signals(date)
                    for name, sig in signals.items():
                        if sig is None:
                            continue
                        # Queue only if: no open trade in this sleeve + not already pending
                        if (not self.risk.sleeve_has_open(sleeve, name)
                                and name not in self.pending[sleeve]):
                            self.pending[sleeve][name] = sig

            # ── 4. MTM snapshot + HWM / pause gate ───────────────────────────
            total_val = self.risk.mark_to_market(d1_bars)
            hwm = max(hwm, total_val)
            if self.dd_pause_pct > 0.0 and hwm > 0:
                current_dd = (hwm - total_val) / hwm
                dd_paused  = current_dd > self.dd_pause_pct

            self._snapshots.append(DailySnapshot(
                date         = date,
                equity       = self.risk.equity,
                total_value  = total_val,
                open_trades  = len(self.risk.all_open()),
                total_r_open = self.risk.open_r(),
                new_trades   = new_today,
                closed_trades= closed_today,
            ))

        # Close remaining open trades at last bar
        last_date = pd.Timestamp(trading_days[-1])
        last_bars: Dict[str, pd.Series] = {}
        for name in self.active_instruments:
            d1_df = self._d1_indicators[_ref_sleeve][name]
            if last_date in d1_df.index:
                last_bars[name] = d1_df.loc[last_date]
        self.risk.close_remaining(last_date, last_bars)

        n     = len(self.risk.closed_trades)
        parts = "  ".join(
            f"{s}:{sum(1 for t in self.risk.closed_trades if t.sleeve == s)}"
            for s in self.risk.sleeves
        )
        print(f"Backtest complete. {n} trades total  ({parts})")
        return self._equity_curve_df()

    # ── Output helpers ────────────────────────────────────────────────────────

    def _equity_curve_df(self) -> pd.DataFrame:
        rows = [{
            "date":          s.date,
            "equity":        s.equity,
            "total_value":   s.total_value,
            "open_trades":   s.open_trades,
            "total_r_open":  s.total_r_open,
            "new_trades":    s.new_trades,
            "closed_trades": s.closed_trades,
        } for s in self._snapshots]
        return pd.DataFrame(rows).set_index("date")

    def get_sector_cap_rejections(self) -> Dict[str, int]:
        """Return per-sector count of trades rejected by the sector cap gate."""
        return dict(self.risk.sector_cap_rejections)

    def get_trades_df(self) -> pd.DataFrame:
        rows = []
        for t in self.risk.closed_trades:
            rows.append({
                "trade_id":     t.trade_id,
                "sleeve":       t.sleeve,
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
                "entry_equity":    t.entry_equity,
                "risk_multiplier": t.risk_multiplier,
                "pyramid_added":   t.pyramid_added,
                "hold_days":       (t.exit_date - t.entry_date).days
                                   if t.exit_date else None,
            })
        return pd.DataFrame(rows)
