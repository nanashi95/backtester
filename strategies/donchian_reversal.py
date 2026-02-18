"""
donchian_reversal.py
--------------------
Pure breakout reversal strategy — Donchian 50-day entry/exit.

Entry:
  - Long:  daily close > highest high of previous 50 days
  - Short: daily close < lowest low of previous 50 days
  - Execute at next daily open

Exit:
  - ONLY on opposite 50-day breakout (reversal signal)
  - No 20-day channel exit
  - No trailing stop ratchet

Stop:
  - Initial disaster stop = 2 x ATR(20), never moves
  - Risk per trade = 0.7% of equity
"""

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyConfig


ENTRY_LOOKBACK = 50
ATR_PERIOD     = 20


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class DonchianReversal(Strategy):
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="Donchian(50) Reversal Only",
            atr_period=ATR_PERIOD,
            atr_initial_stop=2.0,
            atr_trailing_stop=float("inf"),   # disable trailing ratchet
            use_reversal_exit=True,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["atr"] = _atr(d1, ATR_PERIOD)

        # 50-day channel — previous N days (shift(1))
        d1["hh50"] = d1["high"].rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).max().shift(1)
        d1["ll50"] = d1["low"].rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).min().shift(1)

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
