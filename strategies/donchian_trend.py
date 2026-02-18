"""
donchian_trend.py
-----------------
Donchian(50) Reversal + SMA(200) trend filter.

Identical to DonchianReversal except entries are filtered:
  - Long  only when close > SMA(200)
  - Short only when close < SMA(200)

Everything else unchanged:
  - Exit on opposite 50-day breakout
  - Disaster stop = 2 × ATR(20), never moves
  - Risk per trade = 0.7% of equity
"""

from typing import Optional

import pandas as pd

from strategies.base import Strategy, StrategyConfig


ENTRY_LOOKBACK = 50
ATR_PERIOD     = 20
TREND_PERIOD   = 200


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class DonchianTrend(Strategy):
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="Donchian(50) + SMA(200) Filter",
            atr_period=ATR_PERIOD,
            atr_initial_stop=2.0,
            atr_trailing_stop=float("inf"),
            use_reversal_exit=True,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["atr"]   = _atr(d1, ATR_PERIOD)
        d1["sma200"] = d1["close"].rolling(TREND_PERIOD, min_periods=TREND_PERIOD).mean()
        d1["hh50"]   = d1["high"].rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).max().shift(1)
        d1["ll50"]   = d1["low"].rolling(ENTRY_LOOKBACK,  min_periods=ENTRY_LOOKBACK).min().shift(1)
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        hh50   = d1_row.get("hh50")
        ll50   = d1_row.get("ll50")
        atr    = d1_row.get("atr")
        sma200 = d1_row.get("sma200")

        if pd.isna(hh50) or pd.isna(ll50) or pd.isna(atr) or pd.isna(sma200):
            return None

        close = d1_row["close"]

        if close > hh50 and close > sma200:   # long: breakout + above trend
            return 1
        if close < ll50 and close < sma200:   # short: breakdown + below trend
            return -1
        return None
