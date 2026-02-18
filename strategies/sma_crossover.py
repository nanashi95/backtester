"""
sma_crossover.py
----------------
SMA(50)/SMA(200) golden/death crossover strategy.

Signals:
  - SMA50 crosses above SMA200 → LONG  (+1)
  - SMA50 crosses below SMA200 → SHORT (-1)

Risk parameters:
  - ATR(20) for stop/sizing
  - Initial stop = 2 x ATR
  - Trailing stop = 2 x ATR
  - Reversal exit enabled
"""

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyConfig


SMA_FAST   = 50
SMA_SLOW   = 200
ATR_PERIOD = 20


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """True Range -> ATR(period)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class SmaCrossover(Strategy):
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="SMA(50/200) Crossover",
            atr_period=ATR_PERIOD,
            atr_initial_stop=2.0,
            atr_trailing_stop=2.0,
            use_reversal_exit=True,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["sma50"]  = _sma(d1["close"], SMA_FAST)
        d1["sma200"] = _sma(d1["close"], SMA_SLOW)
        d1["atr"]    = _atr(d1, ATR_PERIOD)

        # Crossover detection
        d1["cross_up"] = (
            (d1["sma50"] > d1["sma200"]) &
            (d1["sma50"].shift(1) <= d1["sma200"].shift(1))
        )
        d1["cross_down"] = (
            (d1["sma50"] < d1["sma200"]) &
            (d1["sma50"].shift(1) >= d1["sma200"].shift(1))
        )
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        if pd.isna(d1_row.get("sma200")) or pd.isna(d1_row.get("atr")):
            return None
        if d1_row["cross_up"]:
            return 1
        if d1_row["cross_down"]:
            return -1
        return None
