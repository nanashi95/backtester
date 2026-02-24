"""
ema_trend_fx.py
---------------
EMA50/EMA200 trend-following strategy for FX instruments.

Signal:
  Long  when EMA50 crosses above EMA200 (signal on day T → execute T+1 open)
  Short when EMA50 crosses below EMA200
  Re-entry only on a new cross (stops out within the same trend = flat until next cross)

Risk (identical to Donchian sleeves):
  Initial stop:   1.5 × ATR(20)
  Trailing stop:  2.0 × ATR(20)
  No reversal exit — ATR trailing stop manages all exits
"""

from typing import Optional
import pandas as pd
from strategies.base import Strategy, StrategyConfig


ATR_PERIOD   = 20
ATR_INITIAL  = 1.5
ATR_TRAILING = 2.0
EMA_FAST     = 50
EMA_SLOW     = 200


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class EMATrendFX(Strategy):
    """
    EMA crossover trend strategy.
    Signal fires only on the cross day; ATR trailing stop manages exits.
    Compatible with the ensemble engine's D1 signal queue.
    """

    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name              = f"EMA({EMA_FAST}/{EMA_SLOW}) Trend",
            atr_period        = ATR_PERIOD,
            atr_initial_stop  = ATR_INITIAL,
            atr_trailing_stop = ATR_TRAILING,
            use_reversal_exit = False,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["atr"]    = _atr(d1, ATR_PERIOD)
        d1["ema50"]  = d1["close"].ewm(
            span=EMA_FAST,  adjust=False, min_periods=EMA_FAST).mean()
        d1["ema200"] = d1["close"].ewm(
            span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()

        above      = (d1["ema50"] > d1["ema200"])
        prev_above = above.shift(1).fillna(False).astype(bool)

        sig = pd.Series(0, index=d1.index, dtype=int)
        sig[ above & ~prev_above] =  1   # cross up  → long signal
        sig[~above &  prev_above] = -1   # cross down → short signal
        d1["ema_signal"] = sig
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        sig = d1_row.get("ema_signal")
        if pd.isna(sig) or int(sig) == 0:
            return None
        atr = d1_row.get("atr")
        if pd.isna(atr) or float(atr) <= 0:
            return None
        return int(sig)
