"""
real_data_loader.py
-------------------
Loads real D1 OHLC data from yfinance for all 16 instruments (2008-2024).
H4 bars are synthesised from real daily bars (6 bars per day).

Drop-in replacement for data_generator.py — exposes the same
load_all_data() and get_instrument_bucket() interface.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

# ── Instrument → yfinance ticker mapping ──────────────────────────────────────
TICKER_MAP = {
    "US100":  "^NDX",
    "US500":  "^GSPC",
    "JP225":  "^N225",
    "AUDJPY": "AUDJPY=X",
    "EURJPY": "EURJPY=X",
    "USDJPY": "JPY=X",
    "CADJPY": "CADJPY=X",
    "NZDJPY": "NZDJPY=X",
    "UKOil":  "BZ=F",
    "USOil":  "CL=F",
    "Gold":   "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Sugar":  "SB=F",
    "Coffee": "KC=F",
    "Cocoa":  "CC=F",
}

BUCKET_MAP = {
    "US100":  "equity",
    "US500":  "equity",
    "JP225":  "equity",
    "AUDJPY": "fx_yen",
    "EURJPY": "fx_yen",
    "USDJPY": "fx_yen",
    "CADJPY": "fx_yen",
    "NZDJPY": "fx_yen",
    "Gold":   "metals",
    "Silver": "metals",
    "Copper": "metals",
    "UKOil":  "energy",
    "USOil":  "energy",
    "Sugar":  "softs",
    "Coffee": "softs",
    "Cocoa":  "softs",
}


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_d1.csv")


def _load_cached_d1(name: str) -> Optional[pd.DataFrame]:
    """Load D1 data from CSV cache. Returns None if not cached."""
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def _save_cached_d1(name: str, df: pd.DataFrame) -> None:
    """Save D1 data to CSV cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(_cache_path(name))


def _download_d1(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from yfinance."""
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        return df

    # yfinance may return MultiIndex columns for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = None
    return df


def _synthesise_h4(d1: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Build 6 H4 bars per D1 bar from real daily OHLC.
    Intraday path is a scaled random walk anchored to daily open/close,
    constrained within daily high/low.
    """
    rng = np.random.default_rng(seed)
    h4_bars = []

    for i in range(len(d1)):
        row = d1.iloc[i]
        day_open  = row["open"]
        day_close = row["close"]
        day_high  = row["high"]
        day_low   = row["low"]
        day_date  = d1.index[i]

        # Build intraday path: 7 anchor points (open + 6 bar closes)
        daily_vol = abs(day_close - day_open) / day_open if day_open != 0 else 0.01
        daily_vol = max(daily_vol, 0.001)

        intra_path = np.zeros(7)
        intra_path[0] = day_open
        shocks = rng.normal(0, daily_vol / np.sqrt(6), size=6)
        for j in range(1, 7):
            intra_path[j] = intra_path[j - 1] * np.exp(shocks[j - 1])

        # Scale so last point = day_close
        if intra_path[-1] != 0:
            scale = day_close / intra_path[-1]
            intra_path *= scale

        for b in range(6):
            bar_open  = intra_path[b]
            bar_close = intra_path[b + 1]
            bar_high  = max(bar_open, bar_close) * (1 + abs(rng.normal(0, daily_vol * 0.2)))
            bar_low   = min(bar_open, bar_close) * (1 - abs(rng.normal(0, daily_vol * 0.2)))
            # Constrain within daily range
            bar_high = min(bar_high, day_high)
            bar_low  = max(bar_low, day_low)
            # Ensure OHLC integrity
            bar_high = max(bar_high, bar_open, bar_close)
            bar_low  = min(bar_low, bar_open, bar_close)

            bar_ts = pd.Timestamp(day_date) + pd.Timedelta(hours=b * 4)
            h4_bars.append({
                "timestamp": bar_ts,
                "open":  bar_open,
                "high":  bar_high,
                "low":   bar_low,
                "close": bar_close,
            })

    h4 = pd.DataFrame(h4_bars).set_index("timestamp")
    return h4


def load_all_data(start: str = "2008-01-01",
                  end: str   = "2024-12-31") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Return dict: {instrument_name: {"D1": df, "H4": df}}
    Each df has columns: open, high, low, close (and volume for D1).
    D1 data is real from yfinance. H4 is synthesised from D1.
    """
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    failed = []

    for idx, (name, ticker) in enumerate(TICKER_MAP.items()):
        # Try cache first
        cached = _load_cached_d1(name)
        if cached is not None:
            d1 = cached
            print(f"  Cached  {name:10s} | D1 bars: {len(d1):5d}", end="")
        else:
            print(f"  Downloading {name:10s} ({ticker})...", end="", flush=True)
            d1 = _download_d1(ticker, start, end)

            if d1.empty:
                print(f" FAILED — no data returned")
                failed.append(name)
                continue

            # Drop any rows with NaN OHLC
            d1 = d1.dropna(subset=["open", "high", "low", "close"])
            _save_cached_d1(name, d1)

        h4 = _synthesise_h4(d1, seed=idx * 100)

        all_data[name] = {"D1": d1, "H4": h4}
        print(f" | H4 bars: {len(h4):6d}")

    if failed:
        print(f"\n  WARNING: Failed to load {len(failed)} instruments: {failed}")

    return all_data


def get_instrument_bucket(name: str) -> str:
    return BUCKET_MAP[name]


if __name__ == "__main__":
    print("Downloading real OHLC data 2008-2024...")
    data = load_all_data()
    for name, frames in data.items():
        d1 = frames["D1"]
        assert (d1["high"] >= d1["close"]).all(), f"{name} H<C violation"
        assert (d1["low"]  <= d1["close"]).all(), f"{name} L>C violation"
        assert (d1["high"] >= d1["low"]).all(),   f"{name} H<L violation"
    print("All OHLC integrity checks passed.")
