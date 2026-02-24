"""
base.py
-------
Abstract strategy interface.

Every strategy must:
  1. Return a StrategyConfig with risk parameters
  2. Pre-compute indicator columns (including standardised 'atr' column)
  3. Return per-bar signals (+1 long, -1 short, or None)

Strategies that use H4 data should override:
  - uses_h4 (return True)
  - precompute_h4(h4) — add H4 indicator columns
  - get_d1_trend(d1_row) — return D1 trend direction
  - get_h4_signal(h4_row, d1_trend) — return H4 entry signal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StrategyConfig:
    name: str                    # Display name, e.g. "SMA(50/200) Crossover"
    atr_period: int              # ATR period (20, 14, etc.)
    atr_initial_stop: float      # Initial stop multiplier (2.0, 1.5, etc.)
    atr_trailing_stop: float     # Trailing stop multiplier (2.0, etc.)
    use_reversal_exit: bool      # Close on opposite signal?


class Strategy(ABC):
    @abstractmethod
    def config(self) -> StrategyConfig:
        """Return strategy configuration / risk parameters."""
        ...

    @abstractmethod
    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to D1. MUST add an 'atr' column."""
        ...

    @abstractmethod
    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        """Return +1 (long), -1 (short), or None (no signal)."""
        ...

    def get_exit_signal(self, date: pd.Timestamp, d1_row: pd.Series,
                        trade_direction: int) -> bool:
        """Return True if an open trade should be exited (e.g. channel exit).
        Default: no strategy-level exit (only trailing stop / reversal)."""
        return False

    def is_trend_intact(self, date: pd.Timestamp, d1_row: pd.Series,
                        direction: int) -> bool:
        """Return True if the trend is still valid for an open position.
        Used for pyramid qualification check. Default: always True."""
        return True

    # ── H4 multi-timeframe support ───────────────────────────────────────

    @property
    def uses_h4(self) -> bool:
        """Whether this strategy uses H4 data for entries/management."""
        return False

    def precompute_h4(self, h4: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to H4. Override if uses_h4 is True.
        MUST add an 'atr' column."""
        return h4

    def get_d1_trend(self, d1_row: pd.Series) -> Optional[int]:
        """Return D1 trend direction: +1 (long bias), -1 (short bias), or None."""
        return None

    def get_h4_signal(self, h4_row: pd.Series, d1_trend: int) -> Optional[int]:
        """Return H4 entry signal filtered by D1 trend: +1, -1, or None."""
        return None
