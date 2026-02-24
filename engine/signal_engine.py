"""
signal_engine.py
----------------
Strategy-agnostic signal adapter.

Takes any Strategy instance and delegates indicator computation and
signal generation to it.  The only contract is:
  - strategy.precompute(d1) returns D1 DataFrame with an 'atr' column
  - strategy.get_signal(date, d1_row) returns +1, -1, or None
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict

from strategies.base import Strategy


@dataclass
class SignalResult:
    instrument:     str
    date:           pd.Timestamp
    direction:      int            # +1, -1
    atr:            float          # D1 ATR for stop/sizing
    entry_price:    float          # price at which trade is entered (next day open)
    d1_close:       float          # D1 close for reference
    atr_percentile: float = float('nan')  # ATR rank vs 504-bar window (0–1); NaN = unscaled


def precompute_indicators(strategy: Strategy,
                          d1: pd.DataFrame,
                          h4: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """
    Pre-compute all indicators via the strategy.
    Returns dict with 'D1' (and optionally 'H4') DataFrames.
    """
    d1 = strategy.precompute(d1)
    result = {"D1": d1}
    if h4 is not None and strategy.uses_h4:
        result["H4"] = strategy.precompute_h4(h4)
    return result


class SignalEngine:
    """
    Event-driven signal engine.  Call .get_signals(date) to retrieve
    signals for all instruments on a given calendar date.
    """

    def __init__(self,
                 strategy: Strategy,
                 all_indicators: Dict[str, Dict[str, pd.DataFrame]],
                 instruments: list):
        self.strategy    = strategy
        self.indicators  = all_indicators   # {name: {"D1": df}}
        self.instruments = instruments

    def get_signals(self, date: pd.Timestamp) -> Dict[str, Optional[SignalResult]]:
        """
        For each instrument, ask the strategy if a signal fires on `date`.
        Returns dict {instrument: SignalResult or None}.
        """
        results: Dict[str, Optional[SignalResult]] = {}

        for name in self.instruments:
            d1 = self.indicators[name]["D1"]

            if date not in d1.index:
                results[name] = None
                continue

            d1_row = d1.loc[date]
            direction = self.strategy.get_signal(date, d1_row)

            if direction is None:
                results[name] = None
                continue

            raw_pct = d1_row.get("atr_pct", float("nan"))
            atr_pct = float(raw_pct) if not pd.isna(raw_pct) else float("nan")

            results[name] = SignalResult(
                instrument     = name,
                date           = date,
                direction      = direction,
                atr            = float(d1_row["atr"]),
                entry_price    = float(d1_row["close"]),
                d1_close       = float(d1_row["close"]),
                atr_percentile = atr_pct,
            )

        return results

    def is_trend_intact(self, name: str, date: pd.Timestamp,
                        direction: int) -> bool:
        """Delegate trend-intact check to the strategy for a specific instrument/date."""
        d1 = self.indicators.get(name, {}).get("D1")
        if d1 is None or date not in d1.index:
            return False
        return self.strategy.is_trend_intact(date, d1.loc[date], direction)

    def check_exit_signal(self, name: str, date: pd.Timestamp,
                          direction: int) -> bool:
        """Return True if the strategy requests an EOD exit for this trade."""
        d1 = self.indicators.get(name, {}).get("D1")
        if d1 is None or date not in d1.index:
            return False
        return self.strategy.get_exit_signal(date, d1.loc[date], direction)
