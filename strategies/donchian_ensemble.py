"""
donchian_ensemble.py
--------------------
Three Donchian breakout sleeves for the 3-horizon ensemble system.

All sleeves share identical risk rules:
  Entry:         close > N-day high (long) or close < N-day low (short)
                 — signal on day T, execute at day T+1 open
  Initial stop:  1.5 × ATR(20) from entry
  Trailing stop: 2.0 × ATR(20), ratchets in the favourable direction
  Exit:          ATR trailing stop only — no reversal exit

Sleeve A  Fast:    Donchian(20)
Sleeve B  Medium:  Donchian(50)
Sleeve C  Slow:    Donchian(100)
"""

from typing import Optional
import numpy as np
import pandas as pd
from strategies.base import Strategy, StrategyConfig


ATR_PERIOD    = 20
ATR_INITIAL   = 1.5
ATR_TRAILING  = 2.0

ADX_PERIOD    = 14
ADX_THRESHOLD = 20.0


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean()


class _DonchianSleeve(Strategy):
    """Parameterised Donchian breakout sleeve."""

    def __init__(self, period: int, label: str):
        self._period = period
        self._label  = label

    def config(self) -> StrategyConfig:
        return StrategyConfig(
            name             = f"Sleeve {self._label} — Donchian({self._period})",
            atr_period       = ATR_PERIOD,
            atr_initial_stop = ATR_INITIAL,
            atr_trailing_stop= ATR_TRAILING,
            use_reversal_exit= False,
        )

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = d1.copy()
        d1["atr"] = _atr(d1, ATR_PERIOD)
        n = self._period
        d1[f"hh{n}"] = d1["high"].rolling(n, min_periods=n).max().shift(1)
        d1[f"ll{n}"] = d1["low"].rolling(n, min_periods=n).min().shift(1)
        d1["atr_pct"] = _atr_percentile(d1, lookback=252, min_periods=126)
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        n   = self._period
        hh  = d1_row.get(f"hh{n}")
        ll  = d1_row.get(f"ll{n}")
        atr = d1_row.get("atr")
        if pd.isna(hh) or pd.isna(ll) or pd.isna(atr):
            return None
        close = d1_row["close"]
        if close > hh:
            return 1
        if close < ll:
            return -1
        return None

    def is_trend_intact(self, date: pd.Timestamp, d1_row: pd.Series,
                        direction: int) -> bool:
        """Trend is intact as long as close has not reversed through the channel."""
        n = self._period
        close = float(d1_row["close"])
        if direction == 1:  # long: close must be above lower channel
            ll = d1_row.get(f"ll{n}")
            return not pd.isna(ll) and close > float(ll)
        else:  # short: close must be below upper channel
            hh = d1_row.get(f"hh{n}")
            return not pd.isna(hh) and close < float(hh)


def _adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    Average Directional Index using Wilder's smoothing (alpha = 1/period).
    Returns ADX as a Series aligned with df's index.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move  = high - prev_high
    dn_move  = prev_low - low
    plus_dm  = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)

    alpha    = 1.0 / period
    sm_tr    = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    sm_pdm   = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    sm_mdm   = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di  = 100.0 * sm_pdm / (sm_tr + 1e-10)
    minus_di = 100.0 * sm_mdm / (sm_tr + 1e-10)
    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx      = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return adx


class DonchianSleeveA(_DonchianSleeve):
    """Fast sleeve — Donchian(20), 1.5/2.0 ATR stops."""
    def __init__(self):
        super().__init__(period=20, label="A")


class DonchianSleeveA_ADX(_DonchianSleeve):
    """
    Fast sleeve — Donchian(20), gated by ADX(14) > 20.
    Identical risk to DonchianSleeveA; skips signals in non-trending conditions.
    """
    def __init__(self):
        super().__init__(period=20, label="A")

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = super().precompute(d1)
        d1["adx14"] = _adx(d1, ADX_PERIOD)
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        adx = d1_row.get("adx14")
        if pd.isna(adx) or float(adx) <= ADX_THRESHOLD:
            return None
        return super().get_signal(date, d1_row)


class DonchianSleeveB(_DonchianSleeve):
    """Medium sleeve — Donchian(50), 1.5/2.0 ATR stops."""
    def __init__(self):
        super().__init__(period=50, label="B")


class DonchianSleeveC(_DonchianSleeve):
    """Slow sleeve — Donchian(100), 1.5/2.0 ATR stops."""
    def __init__(self):
        super().__init__(period=100, label="C")


# ── Volatility-scaled variants ────────────────────────────────────────────────

def _atr_percentile(df: pd.DataFrame, atr_col: str = "atr",
                    lookback: int = 504, min_periods: int = 252) -> pd.Series:
    """
    Rolling percentile rank of ATR vs the past N trading days.
    Returns values in [0, 1].  NaN during warmup (< min_periods).
    """
    return df[atr_col].rolling(lookback, min_periods=min_periods).apply(
        lambda x: float(np.mean(x <= x[-1])), raw=True
    )


# ── SMA(200) persistence gate ─────────────────────────────────────────────────

class _DonchianSleevePersist(_DonchianSleeve):
    """
    Donchian breakout sleeve with SMA(200) persistence gate.

    Entry allowed only when:
      Long:  close > SMA200  AND  SMA200[t] − SMA200[t−20] > 0
      Short: close < SMA200  AND  SMA200[t] − SMA200[t−20] < 0
    Exit logic unchanged.
    """
    _SMA_PERIOD   = 200
    _SLOPE_WINDOW = 20

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = super().precompute(d1)
        d1["sma200"]       = (d1["close"]
                              .rolling(self._SMA_PERIOD, min_periods=self._SMA_PERIOD)
                              .mean())
        d1["sma200_slope"] = d1["sma200"] - d1["sma200"].shift(self._SLOPE_WINDOW)
        return d1

    def get_signal(self, date: pd.Timestamp, d1_row: pd.Series) -> Optional[int]:
        sig = super().get_signal(date, d1_row)
        if sig is None:
            return None

        sma200 = d1_row.get("sma200")
        slope  = d1_row.get("sma200_slope")
        if pd.isna(sma200) or pd.isna(slope):
            return None

        close  = float(d1_row["close"])
        sma200 = float(sma200)
        slope  = float(slope)

        if sig == 1:   # long: above SMA200, slope rising
            return 1 if (close > sma200 and slope > 0) else None
        else:          # short: below SMA200, slope falling
            return -1 if (close < sma200 and slope < 0) else None


class DonchianSleeveBPersist(_DonchianSleevePersist):
    """Donchian(50) + SMA(200) persistence gate."""
    def __init__(self):
        super().__init__(period=50, label="B")


class DonchianSleeveCPersist(_DonchianSleevePersist):
    """Donchian(100) + SMA(200) persistence gate."""
    def __init__(self):
        super().__init__(period=100, label="C")


# ── Volatility-scaled variants ────────────────────────────────────────────────

def make_sleeve(period: int, label: str) -> "_DonchianSleeve":
    """Factory: create a Donchian sleeve with arbitrary period and label."""
    return _DonchianSleeve(period=period, label=label)


class DonchianSleeveBVolScaled(_DonchianSleeve):
    """Medium sleeve — Donchian(50) with ATR percentile column for vol-based risk scaling."""
    def __init__(self):
        super().__init__(period=50, label="B")

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = super().precompute(d1)
        d1["atr_pct"] = _atr_percentile(d1)
        return d1


class DonchianSleeveCVolScaled(_DonchianSleeve):
    """Slow sleeve — Donchian(100) with ATR percentile column for vol-based risk scaling."""
    def __init__(self):
        super().__init__(period=100, label="C")

    def precompute(self, d1: pd.DataFrame) -> pd.DataFrame:
        d1 = super().precompute(d1)
        d1["atr_pct"] = _atr_percentile(d1)
        return d1
