"""
sma_multi_tf.py
---------------
SMA(5/13/48/75) multi-timeframe trend-following strategy.

D1 Trend Filter (entry only):
  - Long bias:  SMA5 > SMA13 > SMA48 > SMA75
  - Short bias: SMA5 < SMA13 < SMA48 < SMA75
  - Ignored once trade is open

H4 Entry:
  - SMA(5) crosses SMA(13) on H4
  - Cross direction must match D1 trend
  - Entry at next H4 open after cross confirmed on H4 close

Stop / Risk:
  - Initial stop = 1.5 x ATR(14) on H4
  - Trailing stop = 2.0 x ATR(14) on H4, updated on H4 close
  - No reversal exit — exits only via stops
"""

from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyConfig


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class SmaMultiTF(Strategy):
    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name="SMA(5/13/48/75) Multi-TF",
            atr_period=14,
            atr_initial_stop=1.5,
            atr_trailing_stop=2.0,
            use_reversal_exit=False,
        )

    @property
    def uses_h4(self) -> bool:
        return True

    # ── D1 indicators (trend filter) ─────────────────────────────────────

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["sma5"]  = _sma(d1["close"], 5)
        d1["sma13"] = _sma(d1["close"], 13)
        d1["sma48"] = _sma(d1["close"], 48)
        d1["sma75"] = _sma(d1["close"], 75)
        d1["atr"]   = _atr(d1, 14)

        d1["trend_long"] = (
            (d1["sma5"] > d1["sma13"]) &
            (d1["sma13"] > d1["sma48"]) &
            (d1["sma48"] > d1["sma75"])
        )
        d1["trend_short"] = (
            (d1["sma5"] < d1["sma13"]) &
            (d1["sma13"] < d1["sma48"]) &
            (d1["sma48"] < d1["sma75"])
        )
        return d1

    # ── H4 indicators (entry signals + ATR for stops) ────────────────────

    def precompute_h4(self, h4: pd.DataFrame) -> pd.DataFrame:
        h4 = h4.copy()
        h4["sma5"]  = _sma(h4["close"], 5)
        h4["sma13"] = _sma(h4["close"], 13)
        h4["atr"]   = _atr(h4, 14)

        # H4 crossover detection
        h4["cross_up"] = (
            (h4["sma5"] > h4["sma13"]) &
            (h4["sma5"].shift(1) <= h4["sma13"].shift(1))
        )
        h4["cross_down"] = (
            (h4["sma5"] < h4["sma13"]) &
            (h4["sma5"].shift(1) >= h4["sma13"].shift(1))
        )
        return h4

    # ── Signal methods ───────────────────────────────────────────────────

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        # Not used — signals come from H4 via get_h4_signal
        return None

    def get_d1_trend(self, d1_row: pd.Series) -> Optional[int]:
        if pd.isna(d1_row.get("sma75")):
            return None
        if d1_row["trend_long"]:
            return 1
        if d1_row["trend_short"]:
            return -1
        return None

    def get_h4_signal(self, h4_row: pd.Series, d1_trend: int) -> Optional[int]:
        if pd.isna(h4_row.get("sma13")) or pd.isna(h4_row.get("atr")):
            return None
        if d1_trend == 1 and h4_row["cross_up"]:
            return 1
        if d1_trend == -1 and h4_row["cross_down"]:
            return -1
        return None
