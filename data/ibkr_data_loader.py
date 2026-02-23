"""
data/ibkr_data_loader.py
------------------------
Load historical D1 OHLC data from data/ibkr/{SYMBOL}_D1.csv files
(produced by ibkr/downloader.py).

Public interface is identical to mt5_data_loader so it can be dropped in
via data/loader.py's configure() call.

CSV format (written by downloader):
    date,open,high,low,close
    2008-01-02,112.89,113.15,112.45,112.73

All instruments are D1-only (no H4).  The engines handle H4=None cleanly.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd

from ibkr.instruments import INSTRUMENTS, BUCKET_MAP

DATA_DIR = Path(__file__).resolve().parent / "ibkr"


def _load_csv(path: Path) -> pd.DataFrame:
    """Parse a single IBKR CSV into a DatetimeIndex DataFrame."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index.name = None
    df = df[["open", "high", "low", "close"]].copy()
    df.sort_index(inplace=True)
    # Remove duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_all_data(start: str = "2008-01-01",
                  end:   str = "2025-12-31") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Return {instrument_name: {"D1": df, "H4": None}} for all available CSVs.

    Instruments with missing CSV files are skipped with a warning.
    """
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    missing = []

    for name in INSTRUMENTS:
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
            warnings.warn(f"  WARNING: {name} has no data in [{start}, {end}]")
            missing.append(name)
            continue

        all_data[name] = {"D1": d1, "H4": None}
        print(f"  Loaded {name:6s} | D1 bars: {len(d1):5d} | "
              f"{d1.index[0].date()} → {d1.index[-1].date()}")

    if missing:
        print(f"\n  INFO: {len(missing)} instrument(s) not loaded "
              f"(CSV missing or empty): {missing}")
        print("  Run:  python3 ibkr/downloader.py --skip-existing\n")

    return all_data


def get_instrument_bucket(name: str) -> str:
    """Return the portfolio bucket for an IBKR instrument name."""
    return BUCKET_MAP.get(name, "unknown")
