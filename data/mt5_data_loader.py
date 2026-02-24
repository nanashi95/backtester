"""
mt5_data_loader.py
------------------
Loads real D1 and H4 OHLC data exported from MT5 for all 16 instruments.

Expected file structure:
    files/data/mt5/
    ├── US100_D1.csv
    ├── US100_H4.csv
    ├── US500_D1.csv
    ├── US500_H4.csv
    └── ...

MT5 CSV format (tab-separated):
    <DATE>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    2008.01.02\t86.150\t86.430\t84.260\t84.730\t48899\t0\t0

Drop-in replacement for data_generator.py — exposes the same
load_all_data() and get_instrument_bucket() interface.
"""

from __future__ import annotations
import os
import pandas as pd
from typing import Dict

MT5_DATA_DIR = os.path.join(os.path.dirname(__file__), "mt5")

INSTRUMENTS = [
    # Equity indices
    "US100", "US500", "US2000", "JP225", "DE30", "GB100", "AU200", "FR40", "EUR50", "US30",
    # FX (yen crosses + majors)
    "AUDJPY", "EURJPY", "USDJPY", "CADJPY", "NZDJPY", "CHFJPY",
    "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "USDNOK",
    # Energy
    "UKOil", "USOil", "NATGAS",
    # Metals
    "Gold", "Silver", "Copper", "PALLAD", "PLATIN",
    # Softs / Agriculture
    "Sugar", "Coffee", "Cocoa", "WHEAT", "SOYBEAN",
    # Crypto
    "BTCUSD", "ETHUSD",
]

BUCKET_MAP = {
    # Equity
    "US100":   "equity",
    "US500":   "equity",
    "US2000":  "equity",
    "JP225":   "equity",
    "DE30":    "equity",
    "GB100":   "equity",
    "AU200":   "equity",
    "FR40":    "equity",
    "EUR50":   "equity",
    "US30":    "equity",
    # FX (yen crosses + majors merged)
    "AUDJPY":  "fx",
    "EURJPY":  "fx",
    "USDJPY":  "fx",
    "CADJPY":  "fx",
    "NZDJPY":  "fx",
    "CHFJPY":  "fx",
    "EURUSD":  "fx",
    "GBPUSD":  "fx",
    "AUDUSD":  "fx",
    "USDCAD":  "fx",
    "USDCHF":  "fx",
    "USDNOK":  "fx",
    # Energy
    "UKOil":   "energy",
    "USOil":   "energy",
    "NATGAS":  "energy",
    # Metals
    "Gold":    "metals",
    "Silver":  "metals",
    "Copper":  "metals",
    "PALLAD":  "metals",
    "PLATIN":  "metals",
    # Softs / Agriculture
    "Sugar":   "softs",
    "Coffee":  "softs",
    "Cocoa":   "softs",
    "WHEAT":   "softs",
    "SOYBEAN": "softs",
    # Crypto
    "BTCUSD":  "crypto",
    "ETHUSD":  "crypto",
}


def _load_mt5_csv(path: str) -> pd.DataFrame:
    """Parse a single MT5 CSV export into a DataFrame with DatetimeIndex.
    Handles both D1 format (DATE only) and intraday format (DATE + TIME)."""
    df = pd.read_csv(path, sep="\t")

    # Build datetime index from DATE + optional TIME
    if "<TIME>" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["<DATE>"] + " " + df["<TIME>"],
            format="%Y.%m.%d %H:%M:%S",
        )
    else:
        df["datetime"] = pd.to_datetime(df["<DATE>"], format="%Y.%m.%d")

    df = df.rename(columns={
        "<OPEN>":  "open",
        "<HIGH>":  "high",
        "<LOW>":   "low",
        "<CLOSE>": "close",
    })
    df.index = df["datetime"]
    df.index.name = None
    # Drop duplicate timestamps, keep last
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close"]]


def load_all_data(start: str = "2008-01-01",
                  end: str   = "2024-12-31") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Return dict: {instrument_name: {"D1": df, "H4": df}}
    Each df has columns: open, high, low, close with DatetimeIndex.
    """
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    failed = []

    for name in INSTRUMENTS:
        d1_path = os.path.join(MT5_DATA_DIR, f"{name}_D1.csv")
        h4_path = os.path.join(MT5_DATA_DIR, f"{name}_H4.csv")

        if not os.path.exists(d1_path):
            print(f"  MISSING  {name:10s} D1 — {d1_path}")
            failed.append(name)
            continue

        d1 = _load_mt5_csv(d1_path)
        d1 = d1.loc[start:end]

        h4 = None
        if os.path.exists(h4_path):
            h4 = _load_mt5_csv(h4_path)
            h4 = h4.loc[start:end]

        all_data[name] = {"D1": d1, "H4": h4}
        h4_label = f"{len(h4):6d}" if h4 is not None else "  n/a"
        print(f"  Loaded {name:10s} | D1 bars: {len(d1):5d} | H4 bars: {h4_label}")

    if failed:
        print(f"\n  WARNING: Missing data for {len(failed)} instruments: {failed}")

    return all_data


def get_instrument_bucket(name: str) -> str:
    return BUCKET_MAP[name]


if __name__ == "__main__":
    print("Loading MT5 data 2008-2024...")
    data = load_all_data()
    for name, frames in data.items():
        d1 = frames["D1"]
        assert (d1["high"] >= d1["close"]).all(), f"{name} H<C violation"
        assert (d1["low"]  <= d1["close"]).all(), f"{name} L>C violation"
        assert (d1["high"] >= d1["low"]).all(),   f"{name} H<L violation"
    print(f"\nAll OHLC integrity checks passed for {len(data)} instruments.")
