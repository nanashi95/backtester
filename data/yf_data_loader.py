"""
data/yf_data_loader.py
----------------------
Load historical D1 OHLC data from data/yfinance/{SYMBOL}_D1.csv files
(produced by yf_downloader.py).

Public interface identical to mt5_data_loader — drop-in via data/loader.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import pandas as pd

from yf_downloader import BUCKET_MAP

DATA_DIR = Path(__file__).resolve().parent / "yfinance"


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index.name = None
    df = df[["open", "high", "low", "close"]].copy()
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df = _clean_ohlc(df)
    return df


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Fix minor OHLC inconsistencies common in Yahoo Finance futures data.

    Yahoo Finance sometimes applies roll adjustments to close but not to OHLC,
    producing close values outside the [low, high] range.  Fix by:
      1. Swap H/L where high < low  (rare, <15 rows across full dataset)
      2. Clamp close to [low, high] (preserves channel levels used by Donchian)
    """
    import numpy as np

    # Fix H < L (swap)
    hl_bad = df["high"] < df["low"]
    if hl_bad.any():
        df.loc[hl_bad, ["high", "low"]] = df.loc[hl_bad, ["low", "high"]].values

    # Clamp close to [low, high]
    df["close"] = df["close"].clip(lower=df["low"], upper=df["high"])

    return df


def load_all_data(start: str = "2000-01-01",
                  end:   str = "2025-12-31") -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {instrument_name: {'D1': df, 'H4': None}} for all available CSVs."""
    from yf_downloader import YF_TICKERS

    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    missing = []

    for name in YF_TICKERS:
        csv_path = DATA_DIR / f"{name}_D1.csv"

        if not csv_path.exists():
            missing.append(name)
            continue

        try:
            d1 = _load_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"  ERROR loading {name}: {exc}")
            missing.append(name)
            continue

        d1 = d1.loc[start:end]

        if d1.empty:
            missing.append(name)
            continue

        all_data[name] = {"D1": d1, "H4": None}
        print(f"  Loaded {name:6s} | D1 bars: {len(d1):5d} | "
              f"{d1.index[0].date()} → {d1.index[-1].date()}")

    if missing:
        print(f"\n  INFO: {len(missing)} instrument(s) not loaded: {missing}")
        print("  Run:  python3 yf_downloader.py\n")

    return all_data


def get_instrument_bucket(name: str) -> str:
    return BUCKET_MAP.get(name, "unknown")
