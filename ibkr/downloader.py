"""
ibkr/downloader.py
------------------
Download historical D1 CONTFUT bars from IB Gateway and save to
data/ibkr/{SYMBOL}_D1.csv.

Pacing:
  IBKR allows 60 historical-data requests per 10 minutes.
  For D1 bars with "30 Y" duration: 1 request per instrument → safe.
  We sleep 12s between requests to stay well within limits.

Prerequisites:
  - IB Gateway running on port 4002 (paper) or 4001 (live)
  - pip install ib_insync

Usage:
  python3 ibkr/downloader.py                  # download all instruments
  python3 ibkr/downloader.py ZN ZB GC         # download specific symbols
  python3 ibkr/downloader.py --skip-existing   # skip already-present CSVs

Output CSV format (standard comma-separated):
  date,open,high,low,close
  2008-01-02,112.89,113.15,112.45,112.73

Note on yield-based contracts (ZN, ZB, ZF):
  These are PRICE-based futures, not yield.  Long = buy duration.
  The signal logic (Donchian breakout) applies without modification.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Resolve project root so imports work regardless of invocation directory ───
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "data" / "ibkr"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PORT_PAPER = 4002
PORT_LIVE  = 4001
CLIENT_ID  = 1
DURATION   = "30 Y"       # 30 years of history
BAR_SIZE   = "1 day"
WHAT       = "TRADES"
USE_RTH    = True          # regular trading hours only
SLEEP_BETWEEN = 12         # seconds between requests (pacing)


def _make_contract(symbol: str, exchange: str, currency: str, sec_type: str):
    """Build an ib_insync Contract object for a continuous futures contract."""
    from ib_insync import Contract
    return Contract(
        symbol   = symbol,
        exchange = exchange,
        currency = currency,
        secType  = sec_type,
    )


def download_symbol(ib, name: str, spec: tuple, output_path: Path) -> bool:
    """
    Download D1 bars for one instrument and save to CSV.

    Args:
        ib:          connected IB instance
        name:        internal instrument name (e.g. "ZN")
        spec:        (symbol, exchange, currency, secType, bucket)
        output_path: destination CSV path

    Returns True on success, False on failure.
    """
    import pandas as pd

    symbol, exchange, currency, sec_type, _ = spec

    print(f"  [{name}]  {symbol} @ {exchange}  ({sec_type}) ...", end=" ", flush=True)

    contract = _make_contract(symbol, exchange, currency, sec_type)

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime     = "",
            durationStr     = DURATION,
            barSizeSetting  = BAR_SIZE,
            whatToShow      = WHAT,
            useRTH          = USE_RTH,
            formatDate      = 1,
            keepUpToDate    = False,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False

    if not bars:
        print("EMPTY — no data returned")
        return False

    df = ib.util.df(bars)

    # Keep only the OHLC columns we need
    df = df[["date", "open", "high", "low", "close"]].copy()

    # Parse dates — IB returns strings like "2008-01-02" for daily bars
    df["date"] = pd.to_datetime(df["date"])

    # Drop non-trading rows (volume=0 stub rows sometimes appear)
    if "volume" in ib.util.df(bars).columns:
        raw = ib.util.df(bars)
        mask = raw["volume"] > 0
        df = df[mask.values].copy()

    df = df.set_index("date")
    df.index.name = "date"
    df.sort_index(inplace=True)

    # Remove duplicate dates (keep last)
    df = df[~df.index.duplicated(keep="last")]

    df.to_csv(output_path, index=True)
    print(f"OK  ({len(df)} bars,  {df.index[0].date()} → {df.index[-1].date()})")
    return True


def run(symbols: list[str] | None = None, skip_existing: bool = False,
        port: int = PORT_PAPER) -> None:
    """
    Main download loop.

    Args:
        symbols:       list of instrument names to download (None = all)
        skip_existing: if True, skip instruments with an existing CSV
        port:          IB Gateway port (4002=paper, 4001=live)
    """
    try:
        import ib_insync
    except ImportError:
        print("ERROR: ib_insync is not installed.  Run:  pip install ib_insync")
        sys.exit(1)

    from ib_insync import IB
    from ibkr.instruments import INSTRUMENTS

    targets = {k: v for k, v in INSTRUMENTS.items()
               if symbols is None or k in symbols}

    if not targets:
        print(f"No matching instruments found for: {symbols}")
        sys.exit(1)

    print(f"\nConnecting to IB Gateway on port {port} ...")
    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=CLIENT_ID)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        print("Make sure IB Gateway is running and API access is enabled.")
        sys.exit(1)

    print(f"Connected.  Downloading {len(targets)} instrument(s) to {DATA_DIR}/\n")

    n_ok = n_skip = n_fail = 0

    for name, spec in targets.items():
        csv_path = DATA_DIR / f"{name}_D1.csv"

        if skip_existing and csv_path.exists():
            print(f"  [{name}]  skipped (file exists)")
            n_skip += 1
            continue

        ok = download_symbol(ib, name, spec, csv_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

        # Pacing — stay under 60 requests / 10 min
        if name != list(targets)[-1]:
            time.sleep(SLEEP_BETWEEN)

    ib.disconnect()

    print(f"\nDone.  OK={n_ok}  skipped={n_skip}  failed={n_fail}")
    if n_fail:
        print("  Re-run failed instruments individually to retry.")


def main():
    parser = argparse.ArgumentParser(
        description="Download IBKR historical D1 CONTFUT data to data/ibkr/."
    )
    parser.add_argument(
        "symbols", nargs="*",
        help="Instrument names to download (e.g. ZN ZB GC).  Default: all."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip instruments that already have a CSV file."
    )
    parser.add_argument(
        "--port", type=int, default=PORT_PAPER,
        help=f"IB Gateway port (default {PORT_PAPER} for paper, {PORT_LIVE} for live)."
    )
    args = parser.parse_args()

    run(
        symbols        = args.symbols or None,
        skip_existing  = args.skip_existing,
        port           = args.port,
    )


if __name__ == "__main__":
    main()
