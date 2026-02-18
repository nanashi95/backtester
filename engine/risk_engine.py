"""
risk_engine.py
--------------
Strategy-agnostic risk management engine.

Position Sizing:
  - Risk per trade = 0.7% of current equity (1R)
  - Stop distance  = config.atr_initial_stop × ATR(D1)
  - Unit size      = (equity × 0.007) / (stop_multiplier × ATR)

Entry Sequencing:
  - Signal queued as pending → executed at next day's D1 open price

Stop Management:
  - Initial stop = entry ± (atr_initial_stop × ATR)
  - Trailing stop = highest/lowest D1 extreme ± (atr_trailing_stop × ATR)
  - Trailing stop only recomputed when highest_fav advances
  - Entry day: highest_fav updated but stop trigger skipped
  - Exit triggered by trailing stop OR reversal (if config.use_reversal_exit)

Portfolio Caps (checked BEFORE any new trade is opened):
  - FX Yen bucket   ≤ 1R open exposure
  - Equity bucket    ≤ 2R open exposure
  - Metals bucket    ≤ 2R open exposure
  - Energy bucket    ≤ 2R open exposure
  - Softs bucket     ≤ 2R open exposure
  - Total portfolio  ≤ 6R open exposure
  - If any cap breached → trade REJECTED (no scaling)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from engine.signal_engine import SignalResult
from strategies.base import StrategyConfig


# ── Risk constants ───────────────────────────────────────────────────────────
RISK_PER_TRADE     = 0.007   # 0.7% of equity = 1R

# Bucket caps (in R units)
CAP_FX_YEN         = 1.0
CAP_OTHER_BUCKET   = 1.0
CAP_TOTAL_PORTFOLIO = 4.0

FX_YEN_INSTRUMENTS = {"AUDJPY", "EURJPY", "USDJPY", "CADJPY", "NZDJPY"}

BUCKET_MAP = {
    "US100":  "equity",
    "US500":  "equity",
    "JP225":  "equity",
    "AUDJPY": "fx_yen",
    "EURJPY": "fx_yen",
    "USDJPY": "fx_yen",
    "CADJPY": "fx_yen",
    "NZDJPY": "fx_yen",
    "Gold":   "metals",
    "Silver": "metals",
    "Copper": "metals",
    "UKOil":  "energy",
    "USOil":  "energy",
    "Sugar":  "softs",
    "Coffee": "softs",
    "Cocoa":  "softs",
}


@dataclass
class Trade:
    trade_id:      int
    instrument:    str
    bucket:        str
    direction:     int        # +1 long, -1 short
    entry_date:    pd.Timestamp
    entry_price:   float
    units:         float      # contract/unit size
    initial_stop:  float
    trailing_stop: float      # current trailing stop level
    highest_fav:   float      # most favourable price seen (for trailing)
    r_risked:      float      # R allocated to this trade (always 1.0)
    entry_equity:  float      # equity at entry

    # Filled on exit
    exit_date:     Optional[pd.Timestamp] = None
    exit_price:    Optional[float] = None
    pnl:           Optional[float] = None
    r_multiple:    Optional[float] = None
    exit_reason:   str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def initial_stop_distance(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def pnl_from_price(self, price: float) -> float:
        return (price - self.entry_price) * self.direction * self.units


@dataclass
class PortfolioState:
    equity:       float
    open_trades:  List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)

    def open_r_exposure(self) -> float:
        return sum(t.r_risked for t in self.open_trades)

    def bucket_r_exposure(self, bucket: str) -> float:
        return sum(t.r_risked for t in self.open_trades if t.bucket == bucket)

    def instrument_has_open_trade(self, instrument: str) -> bool:
        return any(t.instrument == instrument for t in self.open_trades)


class RiskEngine:
    def __init__(self, initial_equity: float = 100_000.0,
                 config: Optional[StrategyConfig] = None):
        self.state = PortfolioState(equity=initial_equity)
        self.config = config
        self._atr_initial_stop  = config.atr_initial_stop  if config else 2.0
        self._atr_trailing_stop = config.atr_trailing_stop if config else 2.0
        self._trade_counter = 0

    # ── Position sizing ──────────────────────────────────────────────────────
    def compute_units(self, equity: float, atr: float) -> float:
        """Risk exactly 1R on the trade via initial stop of N×ATR(D1)."""
        dollar_risk    = equity * RISK_PER_TRADE
        stop_distance  = self._atr_initial_stop * atr
        if stop_distance <= 0:
            return 0.0
        return dollar_risk / stop_distance

    # ── Cap checks ───────────────────────────────────────────────────────────
    def _can_accept_trade(self, signal: SignalResult) -> tuple[bool, str]:
        """Return (allowed, rejection_reason)."""
        state   = self.state
        bucket  = BUCKET_MAP[signal.instrument]

        # Already have open trade in this instrument?
        if state.instrument_has_open_trade(signal.instrument):
            return False, "instrument_already_open"

        # Bucket cap check
        current_bucket_r = state.bucket_r_exposure(bucket)
        cap = CAP_FX_YEN if bucket == "fx_yen" else CAP_OTHER_BUCKET
        if current_bucket_r + 1.0 > cap + 1e-9:
            return False, f"bucket_cap_{bucket}"

        # Total portfolio cap
        if state.open_r_exposure() + 1.0 > CAP_TOTAL_PORTFOLIO + 1e-9:
            return False, "portfolio_cap"

        return True, ""

    # ── Open trade ───────────────────────────────────────────────────────────
    def open_trade(self, signal: SignalResult) -> Optional[Trade]:
        """Attempt to open trade. Returns Trade or None if rejected."""
        allowed, reason = self._can_accept_trade(signal)
        if not allowed:
            return None

        equity = self.state.equity
        units  = self.compute_units(equity, signal.atr)
        if units <= 0:
            return None

        stop_dist     = self._atr_initial_stop * signal.atr
        initial_stop  = signal.entry_price - signal.direction * stop_dist

        self._trade_counter += 1
        trade = Trade(
            trade_id      = self._trade_counter,
            instrument    = signal.instrument,
            bucket        = BUCKET_MAP[signal.instrument],
            direction     = signal.direction,
            entry_date    = signal.date,
            entry_price   = signal.entry_price,
            units         = units,
            initial_stop  = initial_stop,
            trailing_stop = initial_stop,
            highest_fav   = signal.entry_price,
            r_risked      = 1.0,
            entry_equity  = equity,
        )
        self.state.open_trades.append(trade)
        return trade

    # ── Update trailing stops ────────────────────────────────────────────────
    def update_trailing_stops_d1(self,
                                 date: pd.Timestamp,
                                 d1_bars: Dict[str, pd.Series]) -> List[Trade]:
        """
        Update trailing stops once per day using D1 bar price action
        and D1 ATR(20).  The trailing stop only ratchets in the favourable
        direction (no ratchet from ATR contraction alone).
        Entry-day stop trigger is skipped.
        Returns list of trades that hit stop and are now closed.
        """
        closed_today: List[Trade] = []

        for trade in list(self.state.open_trades):
            bar = d1_bars.get(trade.instrument)
            if bar is None:
                continue

            atr_val = bar.get("atr")
            if atr_val is None or pd.isna(atr_val):
                continue

            high = bar["high"]
            low  = bar["low"]
            trail_dist = self._atr_trailing_stop * atr_val

            if trade.direction == 1:
                if high > trade.highest_fav:
                    trade.highest_fav = high
                    new_trail = trade.highest_fav - trail_dist
                    trade.trailing_stop = max(trade.trailing_stop, new_trail)

                # Skip stop trigger on entry day
                if date != trade.entry_date and low <= trade.trailing_stop:
                    self._close_trade(trade, date, trade.trailing_stop, "trailing_stop")
                    closed_today.append(trade)

            else:  # short
                if low < trade.highest_fav:
                    trade.highest_fav = low
                    new_trail = trade.highest_fav + trail_dist
                    trade.trailing_stop = min(trade.trailing_stop, new_trail)

                # Skip stop trigger on entry day
                if date != trade.entry_date and high >= trade.trailing_stop:
                    self._close_trade(trade, date, trade.trailing_stop, "trailing_stop")
                    closed_today.append(trade)

        return closed_today

    # ── Reversal exit ─────────────────────────────────────────────────────────
    def close_on_reversal(self,
                          date: pd.Timestamp,
                          instrument: str,
                          opposite_direction: int,
                          d1_bars: Dict[str, pd.Series]) -> Optional[Trade]:
        """
        Close an open trade if an opposite crossover signal fires.
        Returns closed Trade or None.
        """
        for trade in list(self.state.open_trades):
            if trade.instrument != instrument:
                continue
            if trade.direction == opposite_direction:
                # Same direction — not a reversal
                continue
            bar = d1_bars.get(instrument)
            if bar is None:
                continue
            exit_price = float(bar["close"])
            self._close_trade(trade, date, exit_price, "reversal")
            return trade
        return None

    # ── Close trade ─────────────────────────────────────────────────────────
    def _close_trade(self, trade: Trade,
                     date: pd.Timestamp,
                     exit_price: float,
                     reason: str = "trailing_stop") -> None:
        pnl          = trade.pnl_from_price(exit_price)
        initial_risk = trade.entry_equity * RISK_PER_TRADE
        r_multiple   = pnl / initial_risk if initial_risk > 0 else 0.0

        trade.exit_date   = date
        trade.exit_price  = exit_price
        trade.pnl         = pnl
        trade.r_multiple  = r_multiple
        trade.exit_reason = reason

        self.state.equity += pnl
        self.state.open_trades.remove(trade)
        self.state.closed_trades.append(trade)

    # ── Mark-to-market (for equity curve) ───────────────────────────────────
    def mark_to_market(self, d1_bars: Dict[str, pd.Series]) -> float:
        """Return total portfolio value including open unrealised PnL."""
        mtm = self.state.equity
        for trade in self.state.open_trades:
            bar = d1_bars.get(trade.instrument)
            if bar is not None:
                mtm += trade.pnl_from_price(bar["close"])
        return mtm

    # ── Summary ─────────────────────────────────────────────────────────────
    def get_open_exposure_summary(self) -> Dict[str, float]:
        buckets = {}
        for t in self.state.open_trades:
            buckets[t.bucket] = buckets.get(t.bucket, 0.0) + t.r_risked
        return {
            "total_r":   self.state.open_r_exposure(),
            **{f"{k}_r": v for k, v in buckets.items()},
        }
