"""
tsmom.py
--------
Time-Series Momentum (TSMOM) strategy — Engine 2.

Signal logic:
  LONG  : 50-day return > 0  AND  100-day return > 0
  SHORT : 50-day return < 0  AND  100-day return < 0

  Entry fires on the day the condition BECOMES true (transition).
  Between transitions, no new entry is generated.

Exit logic (strategy-level):
  Long  exits when: 50-day return < 0  OR  100-day return < 0
  Short exits when: 50-day return > 0  OR  100-day return > 0

  ATR trailing stop also applies — whichever fires first closes the trade.

Risk params (same as Breakout ensemble):
  ATR period        : 20
  Initial stop mult : 1.5 × ATR
  Trailing stop mult: 2.0 × ATR
"""

from typing import Optional
import pandas as pd
from strategies.base import Strategy, StrategyConfig


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class TSMOMStrategy(Strategy):
    """Time-Series Momentum on 50-day and 100-day return windows."""

    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name              = "TSMOM(50/100)",
            atr_period        = 20,
            atr_initial_stop  = 1.5,
            atr_trailing_stop = 2.0,
            use_reversal_exit = True,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["atr"]    = _atr(d1, 20)
        d1["ret50"]  = d1["close"].pct_change(50)
        d1["ret100"] = d1["close"].pct_change(100)

        both_pos = (d1["ret50"] > 0) & (d1["ret100"] > 0)
        both_neg = (d1["ret50"] < 0) & (d1["ret100"] < 0)

        # Transition: fire only on the day the condition first becomes true
        prev_pos = both_pos.shift(1).fillna(False).astype(bool)
        prev_neg = both_neg.shift(1).fillna(False).astype(bool)

        sig = pd.Series(0, index=d1.index, dtype=int)
        sig[both_pos & ~prev_pos] =  1
        sig[both_neg & ~prev_neg] = -1

        d1["tsmom_signal"] = sig
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        sig = d1_row.get("tsmom_signal")
        if pd.isna(sig) or int(sig) == 0:
            return None
        atr = d1_row.get("atr")
        if pd.isna(atr) or float(atr) <= 0:
            return None
        return int(sig)

    def get_exit_signal(self, date: pd.Timestamp, d1_row: pd.Series,
                        trade_direction: int) -> bool:
        """
        Exit when EITHER return flips sign relative to trade direction.
        Long:  exit if ret50 < 0 OR ret100 < 0
        Short: exit if ret50 > 0 OR ret100 > 0
        """
        ret50  = d1_row.get("ret50")
        ret100 = d1_row.get("ret100")
        if pd.isna(ret50) or pd.isna(ret100):
            return False
        ret50, ret100 = float(ret50), float(ret100)
        if trade_direction == 1:
            return ret50 < 0 or ret100 < 0
        else:
            return ret50 > 0 or ret100 > 0
