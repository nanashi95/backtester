"""
donchian_breakout.py
--------------------
Structural breakout strategy using Donchian Channels.

Entry:
  - Long:  daily close > highest high of previous 50 days
  - Short: daily close < lowest low of previous 50 days
  - Execute at next daily open

Exit (channel):
  - Exit long:  daily close < lowest low of previous 20 days
  - Exit short: daily close > highest high of previous 20 days
  - Execute at next daily open

Stop / Risk:
  - ATR(20) for stop/sizing
  - Initial stop = 2 x ATR(20)
  - Trailing stop = 2 x ATR(20), ratchet only in favourable direction
  - Reversal exit on opposite 50-day breakout signal
"""

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyConfig


ENTRY_LOOKBACK = 50
EXIT_LOOKBACK  = 20
ATR_PERIOD     = 20


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """True Range -> ATR(period)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class DonchianBreakout(Strategy):
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="Donchian(50/20) Breakout",
            atr_period=ATR_PERIOD,
            atr_initial_stop=2.0,
            atr_trailing_stop=2.0,
            use_reversal_exit=True,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["atr"] = _atr(d1, ATR_PERIOD)

        # Entry channels — highest high / lowest low of PREVIOUS N days
        # shift(1) so we compare today's close against yesterday's channel
        d1["hh50"] = d1["high"].rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).max().shift(1)
        d1["ll50"] = d1["low"].rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).min().shift(1)

        # Exit channels — same logic with 20-day lookback
        d1["hh20"] = d1["high"].rolling(EXIT_LOOKBACK, min_periods=EXIT_LOOKBACK).max().shift(1)
        d1["ll20"] = d1["low"].rolling(EXIT_LOOKBACK, min_periods=EXIT_LOOKBACK).min().shift(1)

        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        hh50 = d1_row.get("hh50")
        ll50 = d1_row.get("ll50")
        atr  = d1_row.get("atr")

        if pd.isna(hh50) or pd.isna(ll50) or pd.isna(atr):
            return None

        close = d1_row["close"]

        if close > hh50:
            return 1
        if close < ll50:
            return -1
        return None

    def get_exit_signal(self, date: pd.Timestamp, d1_row: pd.Series,
                        trade_direction: int) -> bool:
        if trade_direction == 1:
            ll20 = d1_row.get("ll20")
            if pd.isna(ll20):
                return False
            return d1_row["close"] < ll20
        else:
            hh20 = d1_row.get("hh20")
            if pd.isna(hh20):
                return False
            return d1_row["close"] > hh20
