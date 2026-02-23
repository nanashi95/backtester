"""
yf_downloader.py
----------------
Download historical D1 futures data from Yahoo Finance and save to
data/yfinance/{SYMBOL}_D1.csv.

All tickers use the =F suffix (front-month continuous futures).  These are
actual exchange prices (CME/CBOT/NYMEX/ICE), the same instruments traded
on IBKR.  No inversion or adjustment needed.  Use this data for backtesting;
use IB Gateway for live execution — the prices will match.

Coverage:
  Rates:       ZN ZB ZF          (~2000–today)
  Equity:      ES NQ YM RTY      (RTY only from 2017; others ~2000)
  FX:          6E 6J 6A 6C 6B   (~2000–today)
  Metals:      GC SI HG PA PL   (~2000–today)
  Energy:      CL NG             (~2000–today)
  Agriculture: ZW ZS ZC SB KC CC (~2000–today)

Usage:
  python3 yf_downloader.py               # download / refresh all
  python3 yf_downloader.py ZN ZB GC      # specific symbols
  python3 yf_downloader.py --skip-existing
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "data" / "yfinance"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2000-01-01"

# Internal name → Yahoo Finance ticker
YF_TICKERS = {
    # Rates
    "ZN":  "ZN=F",
    "ZB":  "ZB=F",
    "ZF":  "ZF=F",
    # Equity indices
    "ES":  "ES=F",
    "NQ":  "NQ=F",
    "RTY": "RTY=F",
    "YM":  "YM=F",
    # FX futures (actual futures prices, same direction as CME contracts)
    "6E":  "6E=F",   # EUR/USD futures
    "6J":  "6J=F",   # JPY/USD futures (USD per JPY × scale)
    "6A":  "6A=F",   # AUD/USD futures
    "6C":  "6C=F",   # CAD/USD futures
    "6B":  "6B=F",   # GBP/USD futures
    # Metals
    "GC":  "GC=F",
    "SI":  "SI=F",
    "HG":  "HG=F",
    "PA":  "PA=F",
    "PL":  "PL=F",
    # Energy
    "CL":  "CL=F",
    "NG":  "NG=F",
    # Agriculture
    "ZW":  "ZW=F",
    "ZS":  "ZS=F",
    "ZC":  "ZC=F",
    "SB":  "SB=F",
    "KC":  "KC=F",
    "CC":  "CC=F",
}

BUCKET_MAP = {
    "ZN": "rates",   "ZB": "rates",   "ZF": "rates",
    "ES": "equity",  "NQ": "equity",  "RTY": "equity", "YM": "equity",
    "6E": "fx",      "6J": "fx",      "6A": "fx",      "6C": "fx",  "6B": "fx",
    "GC": "metals",  "SI": "metals",  "HG": "metals",  "PA": "metals", "PL": "metals",
    "CL": "energy",  "NG": "energy",
    "ZW": "agriculture", "ZS": "agriculture", "ZC": "agriculture",
    "SB": "agriculture", "KC": "agriculture", "CC": "agriculture",
}


def download_symbol(name: str, ticker: str, output_path: Path) -> bool:
    import yfinance as yf
    import pandas as pd

    print(f"  [{name:5s}]  {ticker:8s} ...", end=" ", flush=True)

    warnings.filterwarnings("ignore")
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    warnings.resetwarnings()

    if df is None or df.empty:
        print("EMPTY — no data returned")
        return False

    # yfinance returns MultiIndex columns when auto_adjust=True in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df.index.name = "date"
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    # Drop rows where all OHLC are NaN (stale/missing bars)
    df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

    df.to_csv(output_path, index=True)
    print(f"OK  ({len(df)} bars,  {df.index[0].date()} → {df.index[-1].date()})")
    return True


def run(symbols: list[str] | None = None, skip_existing: bool = False) -> None:
    try:
        import yfinance
    except ImportError:
        print("ERROR: yfinance not installed.  Run:  pip install yfinance")
        sys.exit(1)

    targets = {k: v for k, v in YF_TICKERS.items()
               if symbols is None or k in symbols}

    if not targets:
        print(f"No matching instruments: {symbols}")
        sys.exit(1)

    print(f"\nDownloading {len(targets)} instrument(s) from Yahoo Finance → {DATA_DIR}/\n")

    n_ok = n_skip = n_fail = 0

    for name, ticker in targets.items():
        csv_path = DATA_DIR / f"{name}_D1.csv"

        if skip_existing and csv_path.exists():
            print(f"  [{name:5s}]  skipped (file exists)")
            n_skip += 1
            continue

        ok = download_symbol(name, ticker, csv_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone.  OK={n_ok}  skipped={n_skip}  failed={n_fail}")


def main():
    parser = argparse.ArgumentParser(
        description="Download historical D1 futures data from Yahoo Finance."
    )
    parser.add_argument(
        "symbols", nargs="*",
        help="Instrument names (e.g. ZN GC 6E).  Default: all."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip instruments that already have a CSV file."
    )
    args = parser.parse_args()
    run(symbols=args.symbols or None, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
