"""
hold_audit.py
─────────────
Diagnose the avg hold_days anomaly in the ensemble backtest.

Expected: ~40-50 days (Donchian D50-150 trend system)
Observed:  11.1 days

Investigation dimensions:
  1. Exit reason breakdown
  2. Hold duration bands (1-2d, 3-5d, 6-10d, 11-20d, 21-40d, 40+d)
  3. Hold days by sleeve (B_ME, C_ME, B_EA, C_EA, B_R, C_R)
  4. Hold days by instrument (sorted shortest → longest)
  5. ATR% at entry (atr / entry_price) for short vs long holds
  6. R-multiple by hold band (are short trades mostly losers?)
  7. Trades per hold band — count and % of total

Usage:
  python3 hold_audit.py                    # runs backtest, saves trades CSV
  python3 hold_audit.py --from-csv         # reads output/hold_audit/trades.csv (skip backtest)
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.loader as loader
import data.yf_data_loader as yf_loader
loader.configure(yf_loader.load_all_data, yf_loader.get_instrument_bucket)

from strategies.donchian_ensemble import make_sleeve
from engine.ensemble_engine import EnsemblePortfolioEngine

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "hold_audit")
TRADES_PATH = os.path.join(OUTPUT_DIR, "trades.csv")

RATES       = ["ZF"]
EQUITY      = ["ES", "NQ", "RTY", "YM"]
METALS      = ["GC", "SI", "HG", "PA", "PL"]
ENERGY      = ["CL", "NG"]
AGRICULTURE = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]
ALL_INSTR   = RATES + EQUITY + METALS + ENERGY + AGRICULTURE

SLEEVE_INSTRUMENTS = {
    "B_ME": METALS + ENERGY,
    "C_ME": METALS + ENERGY,
    "B_EA": EQUITY + AGRICULTURE,
    "C_EA": EQUITY + AGRICULTURE,
    "B_R":  RATES,
    "C_R":  RATES,
}

W = 72

HOLD_BANDS = [
    (1,   2,   "1–2d"),
    (3,   5,   "3–5d"),
    (6,   10,  "6–10d"),
    (11,  20,  "11–20d"),
    (21,  40,  "21–40d"),
    (41,  9999,"41d+"),
]


# ── Run production backtest ───────────────────────────────────────────────────

def run_backtest() -> pd.DataFrame:
    print("Running production backtest (2008–2025)...", flush=True)
    t0 = time.time()
    strategies = {
        "B_ME": make_sleeve(50,  "B"),
        "C_ME": make_sleeve(100, "C"),
        "B_EA": make_sleeve(75,  "B"),
        "C_EA": make_sleeve(150, "C"),
        "B_R":  make_sleeve(50,  "B"),
        "C_R":  make_sleeve(100, "C"),
    }
    eng = EnsemblePortfolioEngine(
        strategies         = strategies,
        start              = "2008-01-01",
        end                = "2025-12-31",
        initial_equity     = 100_000.0,
        instruments        = ALL_INSTR,
        risk_per_trade     = 0.006,
        global_cap         = 6.0,
        sleeve_instruments = SLEEVE_INSTRUMENTS,
        sector_cap_pct     = 0.0,
        use_vol_scaling    = False,
    )
    eng.run()
    trades = eng.get_trades_df()
    print(f"Done in {time.time() - t0:.0f}s  ({len(trades)} trades)", flush=True)
    return trades


# ── Helpers ───────────────────────────────────────────────────────────────────

def _band_label(days: float) -> str:
    for lo, hi, label in HOLD_BANDS:
        if lo <= days <= hi:
            return label
    return "41d+"


def _bar(value: float, total: float, width: int = 30) -> str:
    frac = value / total if total > 0 else 0
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "0.0%"


# ── Analysis sections ─────────────────────────────────────────────────────────

def section_overview(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n{'═'*W}")
    print(f"  HOLD DAYS AUDIT  —  {n} closed trades  (2008–2025 production)")
    print(f"{'═'*W}")
    print(f"  Avg hold (calendar days) : {df['hold_days'].mean():.1f}d")
    print(f"  Median hold              : {df['hold_days'].median():.0f}d")
    print(f"  P10 / P25 / P75 / P90   : "
          f"{np.percentile(df['hold_days'], 10):.0f}d / "
          f"{np.percentile(df['hold_days'], 25):.0f}d / "
          f"{np.percentile(df['hold_days'], 75):.0f}d / "
          f"{np.percentile(df['hold_days'], 90):.0f}d")
    print(f"  Max hold                 : {df['hold_days'].max():.0f}d")
    print(f"  Trades ≤ 5 days          : "
          f"{(df['hold_days'] <= 5).sum()} ({_pct((df['hold_days'] <= 5).sum(), n)})")
    print(f"  Trades ≤ 2 days          : "
          f"{(df['hold_days'] <= 2).sum()} ({_pct((df['hold_days'] <= 2).sum(), n)})")


def section_exit_reasons(df: pd.DataFrame) -> None:
    print(f"\n{'─'*W}")
    print(f"  EXIT REASON BREAKDOWN")
    print(f"{'─'*W}")
    counts = df["exit_reason"].value_counts()
    n = len(df)
    for reason, cnt in counts.items():
        avg_hold = df[df["exit_reason"] == reason]["hold_days"].mean()
        avg_r    = df[df["exit_reason"] == reason]["r_multiple"].mean()
        print(f"  {reason:<22s}  {cnt:>5d}  ({_pct(cnt,n):>6})  "
              f"avg hold: {avg_hold:>5.1f}d  avg R: {avg_r:>+.3f}R")


def section_hold_bands(df: pd.DataFrame) -> None:
    print(f"\n{'─'*W}")
    print(f"  HOLD DURATION BANDS")
    print(f"{'─'*W}")
    n = len(df)
    print(f"  {'Band':<8}  {'N':>5}  {'%Tot':>6}  {'WR':>6}  "
          f"{'AvgR':>7}  {'TotR':>8}  {'Bar'}")
    print(f"  {'─'*65}")
    for lo, hi, label in HOLD_BANDS:
        mask   = (df["hold_days"] >= lo) & (df["hold_days"] <= hi)
        sub    = df[mask]
        cnt    = len(sub)
        wr     = (sub["r_multiple"] > 0).mean() if cnt > 0 else 0
        avg_r  = sub["r_multiple"].mean()        if cnt > 0 else 0
        tot_r  = sub["r_multiple"].sum()         if cnt > 0 else 0
        bar    = _bar(cnt, n, 24)
        print(f"  {label:<8}  {cnt:>5}  {_pct(cnt,n):>6}  {wr:>5.0%}  "
              f"{avg_r:>+.3f}R  {tot_r:>+7.1f}R  {bar}")


def section_by_sleeve(df: pd.DataFrame) -> None:
    print(f"\n{'─'*W}")
    print(f"  HOLD DAYS BY SLEEVE")
    print(f"{'─'*W}")
    print(f"  {'Sleeve':<8}  {'N':>5}  {'AvgHold':>8}  {'MedHold':>8}  "
          f"{'≤5d%':>6}  {'AvgR':>7}  {'TotR':>8}")
    print(f"  {'─'*60}")
    for sleeve in sorted(df["sleeve"].unique()):
        sub  = df[df["sleeve"] == sleeve]
        n    = len(sub)
        avg  = sub["hold_days"].mean()
        med  = sub["hold_days"].median()
        le5  = _pct((sub["hold_days"] <= 5).sum(), n)
        avgr = sub["r_multiple"].mean()
        totr = sub["r_multiple"].sum()
        print(f"  {sleeve:<8}  {n:>5}  {avg:>7.1f}d  {med:>7.0f}d  "
              f"{le5:>6}  {avgr:>+.3f}R  {totr:>+7.1f}R")


def section_by_instrument(df: pd.DataFrame) -> None:
    print(f"\n{'─'*W}")
    print(f"  HOLD DAYS BY INSTRUMENT  (sorted by avg hold, ascending)")
    print(f"{'─'*W}")
    print(f"  {'Instrument':<10}  {'Bucket':<12}  {'N':>5}  {'AvgHold':>8}  "
          f"{'MedHold':>8}  {'≤5d%':>6}  {'AvgR':>7}  {'TotR':>8}")
    print(f"  {'─'*72}")

    rows = []
    for instr in df["instrument"].unique():
        sub  = df[df["instrument"] == instr]
        n    = len(sub)
        avg  = sub["hold_days"].mean()
        med  = sub["hold_days"].median()
        le5  = (sub["hold_days"] <= 5).sum() / n * 100
        avgr = sub["r_multiple"].mean()
        totr = sub["r_multiple"].sum()
        bucket = sub["bucket"].iloc[0] if "bucket" in sub.columns else "?"
        rows.append((avg, instr, bucket, n, avg, med, le5, avgr, totr))

    for _, instr, bucket, n, avg, med, le5, avgr, totr in sorted(rows):
        print(f"  {instr:<10}  {bucket:<12}  {n:>5}  {avg:>7.1f}d  {med:>7.0f}d  "
              f"{le5:>5.1f}%  {avgr:>+.3f}R  {totr:>+7.1f}R")


def section_short_trade_deep_dive(df: pd.DataFrame) -> None:
    """Characterise the ≤5 day trades specifically."""
    print(f"\n{'─'*W}")
    print(f"  SHORT-HOLD DEEP DIVE  (≤5 calendar days)")
    print(f"{'─'*W}")

    short = df[df["hold_days"] <= 5].copy()
    n_short = len(short)
    n_total = len(df)

    if n_short == 0:
        print("  No trades ≤5 days.")
        return

    print(f"  Count: {n_short} ({_pct(n_short, n_total)} of all trades)")
    print(f"  Avg R: {short['r_multiple'].mean():+.3f}R")
    print(f"  Win rate: {(short['r_multiple'] > 0).mean():.1%}")
    print(f"  Total R contribution: {short['r_multiple'].sum():+.1f}R")
    print(f"  Avg hold of these trades: {short['hold_days'].mean():.1f}d")

    print(f"\n  By sleeve:")
    for sleeve in sorted(short["sleeve"].unique()):
        sub = short[short["sleeve"] == sleeve]
        print(f"    {sleeve:<8}  {len(sub):>5} trades  "
              f"avg R {sub['r_multiple'].mean():+.3f}R  "
              f"avg hold {sub['hold_days'].mean():.1f}d")

    print(f"\n  By bucket:")
    for bucket in sorted(short["bucket"].unique()):
        sub = short[short["bucket"] == bucket]
        print(f"    {bucket:<12}  {len(sub):>5} trades  "
              f"avg R {sub['r_multiple'].mean():+.3f}R")

    print(f"\n  Exit reasons for ≤5d trades:")
    for reason, cnt in short["exit_reason"].value_counts().items():
        print(f"    {reason:<22s}  {cnt:>5}  ({_pct(cnt, n_short)})")


def section_r_by_hold_band(df: pd.DataFrame) -> None:
    """R-multiple statistics for each hold band."""
    print(f"\n{'─'*W}")
    print(f"  R-MULTIPLE QUALITY BY HOLD BAND")
    print(f"{'─'*W}")
    print(f"  {'Band':<8}  {'N':>5}  {'WR':>6}  {'AvgW':>7}  "
          f"{'AvgL':>7}  {'PF':>6}  {'TotR':>8}")
    print(f"  {'─'*58}")
    for lo, hi, label in HOLD_BANDS:
        mask  = (df["hold_days"] >= lo) & (df["hold_days"] <= hi)
        sub   = df[mask]
        if len(sub) == 0:
            continue
        wins   = sub[sub["r_multiple"] > 0]["r_multiple"]
        losses = sub[sub["r_multiple"] <= 0]["r_multiple"]
        wr     = len(wins) / len(sub)
        avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l  = losses.mean() if len(losses) > 0 else 0.0
        pf     = (wins.sum() / abs(losses.sum())
                  if losses.sum() != 0 else float("inf"))
        tot_r  = sub["r_multiple"].sum()
        print(f"  {label:<8}  {len(sub):>5}  {wr:>5.0%}  "
              f"{avg_w:>+.3f}R  {avg_l:>+.3f}R  {pf:>6.2f}  {tot_r:>+7.1f}R")


def section_annual_hold(df: pd.DataFrame) -> None:
    """Avg hold per year — shows if anomaly is recent or always present."""
    print(f"\n{'─'*W}")
    print(f"  AVG HOLD DAYS BY YEAR")
    print(f"{'─'*W}")
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["entry_date"]).dt.year
    stats = (df2.groupby("year")
               .agg(n=("hold_days","count"),
                    avg_hold=("hold_days","mean"),
                    avg_r=("r_multiple","mean"))
               .reset_index())
    print(f"  {'Year':<6}  {'N':>5}  {'AvgHold':>8}  {'AvgR':>8}")
    print(f"  {'─'*35}")
    for _, row in stats.iterrows():
        print(f"  {int(row['year']):<6}  {int(row['n']):>5}  "
              f"{row['avg_hold']:>7.1f}d  {row['avg_r']:>+.3f}R")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", action="store_true",
                        help="Load trades from saved CSV instead of re-running backtest")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.from_csv:
        if not os.path.exists(TRADES_PATH):
            print(f"ERROR: {TRADES_PATH} not found. Run without --from-csv first.")
            sys.exit(1)
        print(f"Loading trades from {TRADES_PATH}...", flush=True)
        trades = pd.read_csv(TRADES_PATH, parse_dates=["entry_date", "exit_date"])
    else:
        trades = run_backtest()
        trades.to_csv(TRADES_PATH, index=False)
        print(f"Trades saved → {TRADES_PATH}")

    # Drop any rows with null hold_days
    trades = trades[trades["hold_days"].notna()].copy()
    trades["hold_days"] = trades["hold_days"].astype(float)

    # ── Run all sections ──────────────────────────────────────────────────────
    section_overview(trades)
    section_exit_reasons(trades)
    section_hold_bands(trades)
    section_by_sleeve(trades)
    section_by_instrument(trades)
    section_short_trade_deep_dive(trades)
    section_r_by_hold_band(trades)
    section_annual_hold(trades)

    print(f"\n{'═'*W}")
    print(f"  Trades CSV: {TRADES_PATH}")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
