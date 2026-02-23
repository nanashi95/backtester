"""
ibkr/downloader.py
------------------
Download historical D1 bars from IB Gateway and save to
data/ibkr/{SYMBOL}_D1.csv.

Contract types:
  CONTFUT — continuous front-month futures (rates, equity, metals, energy, agri)
  CASH    — FX spot via IDEALPRO (6E, 6A, 6B, 6J, 6C); same Donchian signals
             as FX futures.  6J and 6C are stored inverted (1/price, H↔L swap)
             to match the direction of the corresponding futures contracts.

Pacing:
  IBKR allows 60 historical-data requests per 10 minutes.
  We sleep 12s between requests to stay well within limits.

Prerequisites:
  - IB Gateway running on port 4002 (paper) or 4001 (live)
  - pip install ib_insync

Usage:
  python3 ibkr/downloader.py                  # download all instruments
  python3 ibkr/downloader.py ZN ZB GC         # download specific symbols
  python3 ibkr/downloader.py --skip-existing   # skip already-present CSVs

Output CSV format:
  date,open,high,low,close
  2008-01-02,112.89,113.15,112.45,112.73
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

PORT_PAPER    = 4002
PORT_LIVE     = 4001
CLIENT_ID     = 12       # default; use --client-id to override if taken
DURATION      = "30 Y"
BAR_SIZE      = "1 day"
USE_RTH       = True
SLEEP_BETWEEN = 12       # seconds between requests


def _make_contract(symbol: str, exchange: str, currency: str, sec_type: str):
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

    spec = (ibkr_symbol, exchange, currency, secType, bucket, invert)
      invert=True: store 1/price with H↔L swap (used for 6J, 6C)
    """
    import pandas as pd

    symbol, exchange, currency, sec_type, _, invert = spec

    # FX CASH uses MIDPOINT; futures use TRADES
    what_to_show = "MIDPOINT" if sec_type == "CASH" else "TRADES"

    label = f"{symbol}/{currency}" if sec_type == "CASH" else f"{symbol} @ {exchange}"
    print(f"  [{name}]  {label}  ({sec_type}) ...", end=" ", flush=True)

    contract = _make_contract(symbol, exchange, currency, sec_type)

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime    = "",
            durationStr    = DURATION,
            barSizeSetting = BAR_SIZE,
            whatToShow     = what_to_show,
            useRTH         = USE_RTH,
            formatDate     = 1,
            keepUpToDate   = False,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False

    if not bars:
        print("EMPTY — no data returned")
        return False

    # Build DataFrame from BarData objects
    records = [
        {
            "date":   b.date,
            "open":   b.open,
            "high":   b.high,
            "low":    b.low,
            "close":  b.close,
            "volume": getattr(b, "volume", -1),
        }
        for b in bars
    ]
    df = pd.DataFrame(records)

    # Drop non-trading rows (volume=0 stubs; skip check for CASH which has no volume)
    if sec_type != "CASH" and df["volume"].gt(0).any():
        df = df[df["volume"] > 0].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.name = "date"
    df = df[["open", "high", "low", "close"]]
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    # Inversion: 1/price with H↔L swap (for 6J, 6C to match futures direction)
    if invert:
        df = pd.DataFrame({
            "open":  1.0 / df["open"],
            "high":  1.0 / df["low"],    # low becomes high after inversion
            "low":   1.0 / df["high"],   # high becomes low after inversion
            "close": 1.0 / df["close"],
        }, index=df.index)

    df.to_csv(output_path, index=True)
    inv_note = "  [inverted]" if invert else ""
    print(f"OK  ({len(df)} bars,  {df.index[0].date()} → {df.index[-1].date()}){inv_note}")
    return True


def run(symbols: list[str] | None = None, skip_existing: bool = False,
        port: int = PORT_PAPER, client_id: int = CLIENT_ID) -> None:
    try:
        import ib_insync
    except ImportError:
        print("ERROR: ib_insync not installed.  Run:  pip install ib_insync")
        sys.exit(1)

    from ib_insync import IB
    from ibkr.instruments import INSTRUMENTS

    targets = {k: v for k, v in INSTRUMENTS.items()
               if symbols is None or k in symbols}

    if not targets:
        print(f"No matching instruments found for: {symbols}")
        sys.exit(1)

    print(f"\nConnecting to IB Gateway on port {port}  (clientId={client_id}) ...")
    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=client_id)
    except Exception as exc:
        print(f"Connection failed: {exc}")
        print("Make sure IB Gateway is running and API access is enabled.")
        print("If 'clientId already in use', try --client-id <other_number>.")
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

        if name != list(targets)[-1]:
            time.sleep(SLEEP_BETWEEN)

    ib.disconnect()

    print(f"\nDone.  OK={n_ok}  skipped={n_skip}  failed={n_fail}")
    if n_fail:
        print("  Re-run failed instruments individually to retry.")


def main():
    parser = argparse.ArgumentParser(
        description="Download IBKR historical D1 data to data/ibkr/."
    )
    parser.add_argument(
        "symbols", nargs="*",
        help="Instrument names to download (e.g. ZN ZB GC 6E).  Default: all."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip instruments that already have a CSV file."
    )
    parser.add_argument(
        "--port", type=int, default=PORT_PAPER,
        help=f"IB Gateway port (default {PORT_PAPER} paper, {PORT_LIVE} live)."
    )
    parser.add_argument(
        "--client-id", type=int, default=CLIENT_ID,
        help=f"API client ID (default {CLIENT_ID})."
    )
    args = parser.parse_args()

    run(
        symbols       = args.symbols or None,
        skip_existing = args.skip_existing,
        port          = args.port,
        client_id     = args.client_id,
    )


if __name__ == "__main__":
    main()
