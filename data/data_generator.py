"""
data_generator.py
-----------------
Generates realistic synthetic OHLC data for all 16 instruments over 2008-2024.
Uses GBM with regime-switching, fat tails, and instrument-specific characteristics.
Replace this module with real broker data (MT4/5 CSV export) for live backtesting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class InstrumentSpec:
    name: str
    bucket: str
    annual_vol: float        # annualised daily vol
    annual_drift: float      # long-run drift
    mean_daily_range: float  # avg H-L as fraction of price
    base_price: float        # starting price
    pip_size: float          # minimum price increment
    spread_pips: float       # typical bid-ask spread


INSTRUMENT_SPECS: Dict[str, InstrumentSpec] = {
    "US100": InstrumentSpec("US100", "equity", 0.22, 0.07, 0.012, 2000.0, 0.1, 2.0),
    "US500": InstrumentSpec("US500", "equity", 0.18, 0.06, 0.010, 1400.0, 0.1, 0.5),
    "JP225": InstrumentSpec("JP225", "equity", 0.20, 0.04, 0.011, 13000.0, 1.0, 10.0),
    "AUDJPY": InstrumentSpec("AUDJPY", "fx_yen", 0.12, 0.01, 0.006, 85.0, 0.001, 0.02),
    "EURJPY": InstrumentSpec("EURJPY", "fx_yen", 0.10, 0.01, 0.005, 160.0, 0.001, 0.02),
    "USDJPY": InstrumentSpec("USDJPY", "fx_yen", 0.09, 0.01, 0.004, 110.0, 0.001, 0.015),
    "CADJPY": InstrumentSpec("CADJPY", "fx_yen", 0.10, 0.01, 0.005, 90.0, 0.001, 0.02),
    "NZDJPY": InstrumentSpec("NZDJPY", "fx_yen", 0.12, 0.01, 0.006, 75.0, 0.001, 0.025),
    "UKOil":  InstrumentSpec("UKOil",  "commodity", 0.35, 0.02, 0.020, 80.0, 0.01, 0.05),
    "USOil":  InstrumentSpec("USOil",  "commodity", 0.36, 0.02, 0.021, 75.0, 0.01, 0.05),
    "Gold":   InstrumentSpec("Gold",   "commodity", 0.18, 0.04, 0.009, 900.0, 0.1, 0.5),
    "Silver": InstrumentSpec("Silver", "commodity", 0.30, 0.03, 0.015, 15.0, 0.001, 0.02),
    "Copper": InstrumentSpec("Copper", "commodity", 0.25, 0.02, 0.013, 3.0, 0.001, 0.005),
    "Sugar":  InstrumentSpec("Sugar",  "commodity", 0.30, 0.01, 0.016, 12.0, 0.001, 0.02),
    "Coffee": InstrumentSpec("Coffee", "commodity", 0.32, 0.01, 0.018, 120.0, 0.01, 0.2),
    "Cocoa":  InstrumentSpec("Cocoa",  "commodity", 0.28, 0.01, 0.015, 2500.0, 1.0, 5.0),
}


def _generate_regime_series(n_days: int, seed_offset: int = 0) -> np.ndarray:
    """Generate regime labels: 0=bull/calm, 1=bear/volatile, 2=crisis."""
    rng = np.random.default_rng(42 + seed_offset)
    regimes = np.zeros(n_days, dtype=int)
    # Markov transitions
    trans = np.array([[0.995, 0.004, 0.001],
                      [0.010, 0.985, 0.005],
                      [0.050, 0.050, 0.900]])
    state = 0
    for i in range(n_days):
        regimes[i] = state
        state = rng.choice(3, p=trans[state])

    # Hard-code 2008-2009 crisis period
    regimes[0:500] = 2
    # 2020 COVID crash
    # roughly day 4400 (2008 + 12 years = 2020)
    crisis_start = int(4400)
    regimes[crisis_start:crisis_start + 90] = 2
    return regimes


def _generate_ohlc(spec: InstrumentSpec,
                   dates_d1: pd.DatetimeIndex,
                   dates_h4: pd.DatetimeIndex,
                   instrument_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate D1 and H4 OHLC for one instrument."""
    rng = np.random.default_rng(instrument_seed)
    n = len(dates_d1)

    regimes = _generate_regime_series(n, instrument_seed)
    vol_mult = np.where(regimes == 2, 3.5, np.where(regimes == 1, 1.8, 1.0))
    drift_mult = np.where(regimes == 2, -2.5, np.where(regimes == 1, -0.5, 1.0))

    daily_vol = spec.annual_vol / np.sqrt(252)
    daily_drift = spec.annual_drift / 252

    # Fat-tailed returns via Student-t
    t_df = 4.5
    raw_shocks = rng.standard_t(t_df, size=n)
    raw_shocks = raw_shocks / np.sqrt(t_df / (t_df - 2))  # normalise variance

    returns = daily_drift * drift_mult + daily_vol * vol_mult * raw_shocks

    # Build close prices
    closes = np.zeros(n)
    closes[0] = spec.base_price
    for i in range(1, n):
        closes[i] = closes[i - 1] * np.exp(returns[i])
        closes[i] = max(closes[i], spec.pip_size)  # floor at pip

    # OHLC construction from close
    mean_range = spec.mean_daily_range * vol_mult
    highs = closes * (1 + mean_range * rng.uniform(0.3, 1.0, n))
    lows  = closes * (1 - mean_range * rng.uniform(0.3, 1.0, n))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    # Ensure H >= max(O,C) and L <= min(O,C)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows, np.minimum(opens, closes))

    d1 = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low":  lows,
        "close": closes,
        "volume": rng.integers(100000, 5000000, size=n).astype(float),
    }, index=dates_d1)

    # ------ H4 data: 6 bars per day (24h market), resample from intraday sim ------
    h4_bars = []
    for i in range(n):
        day_open  = d1.iloc[i]["open"]
        day_close = d1.iloc[i]["close"]
        day_high  = d1.iloc[i]["high"]
        day_low   = d1.iloc[i]["low"]

        # Build 6 H4 closes that end at day_close
        intra_path = np.zeros(7)
        intra_path[0] = day_open
        intra_shocks = rng.normal(0, daily_vol / np.sqrt(6), size=6)
        for j in range(1, 7):
            intra_path[j] = intra_path[j - 1] * np.exp(intra_shocks[j - 1])
        # Scale so last point = day_close
        scale = day_close / intra_path[-1]
        intra_path = intra_path * scale

        day_date = dates_d1[i]
        for b in range(6):
            bar_open  = intra_path[b]
            bar_close = intra_path[b + 1]
            bar_high  = max(bar_open, bar_close) * (1 + abs(rng.normal(0, mean_range[i] * 0.3)))
            bar_low   = min(bar_open, bar_close) * (1 - abs(rng.normal(0, mean_range[i] * 0.3)))
            bar_high  = min(bar_high, day_high)
            bar_low   = max(bar_low, day_low)
            bar_ts    = pd.Timestamp(day_date) + pd.Timedelta(hours=b * 4)
            h4_bars.append({
                "timestamp": bar_ts,
                "open":  bar_open,
                "high":  bar_high,
                "low":   bar_low,
                "close": bar_close,
            })

    h4 = pd.DataFrame(h4_bars).set_index("timestamp")
    # Filter to dates_h4
    h4 = h4[h4.index.isin(dates_h4)] if len(dates_h4) < len(h4) else h4

    return d1, h4


def load_all_data(start: str = "2008-01-01",
                  end: str   = "2024-12-31") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Return dict: {instrument_name: {"D1": df, "H4": df}}
    Each df has columns: open, high, low, close, volume (D1 only).
    """
    dates_d1 = pd.bdate_range(start, end)
    # H4: every 4h for business days
    dates_h4 = pd.date_range(start, end, freq="4h")
    dates_h4 = dates_h4[dates_h4.dayofweek < 5]

    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for idx, (name, spec) in enumerate(INSTRUMENT_SPECS.items()):
        d1, h4 = _generate_ohlc(spec, dates_d1, dates_h4, instrument_seed=idx * 100)
        all_data[name] = {"D1": d1, "H4": h4}
        print(f"  Loaded {name:10s} | D1 bars: {len(d1):5d} | H4 bars: {len(h4):6d}")

    return all_data


def get_instrument_bucket(name: str) -> str:
    return INSTRUMENT_SPECS[name].bucket


if __name__ == "__main__":
    print("Generating synthetic OHLC data 2008-2024...")
    data = load_all_data()
    # Quick sanity check
    for name, frames in data.items():
        d1 = frames["D1"]
        assert (d1["high"] >= d1["close"]).all(), f"{name} H<C violation"
        assert (d1["low"]  <= d1["close"]).all(), f"{name} L>C violation"
        assert (d1["high"] >= d1["low"]).all(),   f"{name} H<L violation"
    print("All OHLC integrity checks passed.")
