"""
analyze_assets.py
-----------------
Per-asset-class trade behaviour analysis for the Donchian(50) Reversal strategy.

Runs a single unconstrained backtest (very high caps) over the FULL instrument
universe so that EVERY signal trades — removing portfolio-selection bias and
giving pure, unfiltered statistics for each asset class.

Metrics computed per bucket:
  1. Average trade duration (mean + median hold days, duration distribution)
  2. Average trend length after breakout (reversal vs stop exits; winner vs loser)
  3. % of trades stopped within first N days (5 / 10 / 20 / 30 / 50)
  4. R-distribution skew (skewness coefficient, percentiles, tail stats)

Usage:
    python3 analyze_assets.py

Output:
    Printed report + output/asset_class_analysis.txt
"""

from __future__ import annotations
import sys, os, io, time
import numpy as np
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(__file__))

from strategies import STRATEGIES
from main import run_strategy
from data.mt5_data_loader import INSTRUMENTS, BUCKET_MAP

# ── Config ────────────────────────────────────────────────────────────────────
START      = "2008-01-01"
END        = "2025-12-31"
EQUITY_K   = 100_000.0
# Very high caps → every signal trades; removes portfolio-selection bias
UNCONSTRAINED_CAP = 100.0   # per bucket and total

# Days thresholds for "stopped within first N days" breakdown
EARLY_STOP_THRESHOLDS = [5, 10, 20, 30, 50]

# Duration brackets for histogram
DURATION_BRACKETS = [
    (1,  5,  "1–5d"),
    (6,  14, "6–14d"),
    (15, 30, "15–30d"),
    (31, 60, "31–60d"),
    (61, 120,"61–120d"),
    (121, 9999, ">120d"),
]

BUCKET_ORDER = ["equity", "fx", "energy", "metals", "softs", "crypto"]


# ── Stats helpers ─────────────────────────────────────────────────────────────

def skewness(arr: np.ndarray) -> float:
    """Fisher-Pearson moment coefficient of skewness."""
    n = len(arr)
    if n < 3:
        return float("nan")
    mu  = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 3) * n * n / ((n - 1) * (n - 2)))


def pct(vals: pd.Series, threshold: float) -> float:
    return 100.0 * (vals <= threshold).sum() / len(vals) if len(vals) > 0 else 0.0


def percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) > 0 else float("nan")


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(trades_df: pd.DataFrame) -> dict:
    """Compute per-bucket statistics. Returns nested dict."""
    results = {}
    all_buckets = trades_df["bucket"].unique()

    for bucket in BUCKET_ORDER:
        if bucket not in all_buckets:
            continue

        df = trades_df[trades_df["bucket"] == bucket].copy()
        n  = len(df)
        if n == 0:
            continue

        hold   = df["hold_days"].values.astype(float)
        r_vals = df["r_multiple"].values.astype(float)
        winners = df[df["r_multiple"] > 0]
        losers  = df[df["r_multiple"] <= 0]
        rev_ex  = df[df["exit_reason"] == "reversal"]
        stop_ex = df[df["exit_reason"] == "trailing_stop"]

        # ── 1. Duration ───────────────────────────────────────────────────────
        duration = {
            "mean":   float(np.mean(hold)),
            "median": float(np.median(hold)),
            "std":    float(np.std(hold, ddof=1)) if n > 1 else 0.0,
            "brackets": {},
        }
        for lo, hi, label in DURATION_BRACKETS:
            cnt = ((hold >= lo) & (hold <= hi)).sum()
            duration["brackets"][label] = (int(cnt), 100.0 * cnt / n)

        # ── 2. Trend length after breakout ────────────────────────────────────
        trend = {
            "n_reversal_exits":  len(rev_ex),
            "n_stop_exits":      len(stop_ex),
            "pct_reversal_exit": 100.0 * len(rev_ex) / n,
            "avg_hold_reversal": float(rev_ex["hold_days"].mean()) if len(rev_ex) else float("nan"),
            "avg_hold_stop":     float(stop_ex["hold_days"].mean()) if len(stop_ex) else float("nan"),
            "avg_hold_winners":  float(winners["hold_days"].mean()) if len(winners) else float("nan"),
            "avg_hold_losers":   float(losers["hold_days"].mean())  if len(losers)  else float("nan"),
        }

        # ── 3. Early stop ─────────────────────────────────────────────────────
        early = {}
        for t in EARLY_STOP_THRESHOLDS:
            all_pct  = pct(df["hold_days"], t)
            loss_pct = pct(losers["hold_days"], t) if len(losers) > 0 else float("nan")
            early[t] = {"all": all_pct, "losers_only": loss_pct}

        # ── 4. R distribution skew ────────────────────────────────────────────
        r_dist = {
            "skewness":     skewness(r_vals),
            "mean":         float(np.mean(r_vals)),
            "median":       float(np.median(r_vals)),
            "std":          float(np.std(r_vals, ddof=1)) if n > 1 else 0.0,
            "p10":          percentile(r_vals, 10),
            "p25":          percentile(r_vals, 25),
            "p75":          percentile(r_vals, 75),
            "p90":          percentile(r_vals, 90),
            "p95":          percentile(r_vals, 95),
            "min":          float(np.min(r_vals)),
            "max":          float(np.max(r_vals)),
            "win_rate":     100.0 * len(winners) / n,
            "avg_win_r":    float(winners["r_multiple"].mean()) if len(winners) else float("nan"),
            "avg_loss_r":   float(losers["r_multiple"].mean())  if len(losers)  else float("nan"),
        }

        results[bucket] = {
            "n_trades": n,
            "instruments": sorted(df["instrument"].unique().tolist()),
            "duration": duration,
            "trend": trend,
            "early_stop": early,
            "r_dist": r_dist,
        }

    return results


# ── Report formatter ──────────────────────────────────────────────────────────

def _bar(pct_val: float, width: int = 20) -> str:
    filled = round(pct_val / 100 * width)
    return "█" * filled + "░" * (width - filled)


def format_report(stats: dict) -> str:
    lines = []

    def L(s=""): lines.append(s)
    def H(title): L(); L("─" * 70); L(f"  {title}"); L("─" * 70)

    L("=" * 70)
    L("  ASSET CLASS TRADE BEHAVIOUR ANALYSIS")
    L("  Donchian(50) Reversal  |  Full universe  |  2008–2025  |  Unconstrained")
    L("=" * 70)

    for bucket, s in stats.items():
        n    = s["n_trades"]
        inst = ", ".join(s["instruments"])
        H(f"{bucket.upper()}  ({n} trades)  ·  {inst}")

        # ── 1. Trade duration ─────────────────────────────────────────────
        d = s["duration"]
        L(f"\n  1. AVERAGE TRADE DURATION")
        L(f"     Mean  : {d['mean']:6.1f} days    Median: {d['median']:5.1f} days    Std: {d['std']:.1f}d")
        L(f"     Duration distribution:")
        for label, (cnt, pct_v) in d["brackets"].items():
            bar = _bar(pct_v, 25)
            L(f"       {label:<10s}  {cnt:4d} trades  ({pct_v:5.1f}%)  {bar}")

        # ── 2. Trend length after breakout ────────────────────────────────
        t = s["trend"]
        L(f"\n  2. TREND LENGTH AFTER BREAKOUT")
        L(f"     Exit by channel reversal : {t['n_reversal_exits']:4d} trades ({t['pct_reversal_exit']:.1f}%)"
          f"  ·  avg hold {t['avg_hold_reversal']:.1f}d")
        L(f"     Exit by disaster stop    : {t['n_stop_exits']:4d} trades ({100-t['pct_reversal_exit']:.1f}%)"
          f"  ·  avg hold {t['avg_hold_stop']:.1f}d")
        L(f"     Winner avg hold          : {t['avg_hold_winners']:.1f}d")
        L(f"     Loser  avg hold          : {t['avg_hold_losers']:.1f}d")
        L(f"     Winner/Loser hold ratio  : {t['avg_hold_winners']/t['avg_hold_losers']:.2f}x"
          if t['avg_hold_losers'] > 0 else "")

        # ── 3. Early stop breakdown ───────────────────────────────────────
        L(f"\n  3. % STOPPED WITHIN FIRST N DAYS")
        L(f"     {'Threshold':<12s}  {'All trades':>10s}  {'Losers only':>12s}")
        L(f"     {'─'*36}")
        for t_days, pcts in s["early_stop"].items():
            bar = _bar(pcts["all"], 15)
            L(f"     ≤{t_days:2d} days     {pcts['all']:8.1f}%  {pcts['losers_only']:10.1f}%    {bar}")

        # ── 4. R distribution ─────────────────────────────────────────────
        r = s["r_dist"]
        L(f"\n  4. R DISTRIBUTION SKEW")
        L(f"     Skewness : {r['skewness']:+.2f}  {'(right-skewed — fat winners)' if r['skewness'] > 0 else '(left-skewed — losses dominate tail)'}")
        L(f"     Win rate : {r['win_rate']:.1f}%    Avg Win: {r['avg_win_r']:.2f}R    Avg Loss: {r['avg_loss_r']:.2f}R")
        L(f"     Mean     : {r['mean']:+.2f}R    Median : {r['median']:+.2f}R")
        L(f"     Percentiles:  P10={r['p10']:+.2f}  P25={r['p25']:+.2f}  P75={r['p75']:+.2f}  P90={r['p90']:+.2f}  P95={r['p95']:+.2f}")
        L(f"     Range    :  Min={r['min']:+.2f}R  Max={r['max']:+.2f}R  Std={r['std']:.2f}")

    L()
    L("=" * 70)

    # ── Cross-bucket comparison ───────────────────────────────────────────────
    L()
    L("=" * 70)
    L("  CROSS-BUCKET SUMMARY")
    L("=" * 70)
    header = (f"  {'Bucket':<10s} {'Trades':>7s} {'MeanHold':>9s} {'MedHold':>8s} "
              f"{'AvgTrend(rev)':>13s} {'Win%':>6s} {'AvgW':>6s} {'AvgL':>6s} "
              f"{'Skew':>6s} {'≤10d%':>6s}")
    L(header)
    L("  " + "─" * 93)
    for bucket, s in stats.items():
        d = s["duration"]
        t = s["trend"]
        r = s["r_dist"]
        early10 = s["early_stop"].get(10, {}).get("all", float("nan"))
        L(f"  {bucket:<10s} {s['n_trades']:>7d} {d['mean']:>9.1f}d {d['median']:>7.1f}d "
          f"  {t['avg_hold_reversal']:>9.1f}d (rev)  {r['win_rate']:>5.1f}% "
          f"{r['avg_win_r']:>5.2f}R {r['avg_loss_r']:>6.2f}R  "
          f"{r['skewness']:>+6.2f}  {early10:>5.1f}%")
    L("=" * 70)

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    strategy = STRATEGIES[0]

    print("=" * 70)
    print("  ASSET CLASS ANALYSIS — generating unconstrained trade data...")
    print(f"  Strategy  : {strategy.config().name}")
    print(f"  Universe  : ALL {len(INSTRUMENTS)} instruments")
    print(f"  Period    : {START} → {END}")
    print(f"  Caps      : {UNCONSTRAINED_CAP}R per bucket / {UNCONSTRAINED_CAP}R total")
    print("=" * 70)

    # Run unconstrained backtest (capture verbose output)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        metrics, equity_curve, trades_df, engine = run_strategy(
            strategy, START, END, EQUITY_K,
            instruments=INSTRUMENTS,
            mode="A",
            cap_per_bucket=UNCONSTRAINED_CAP,
            cap_total=UNCONSTRAINED_CAP,
            silent=True,
        )

    print(f"  {len(trades_df)} trades loaded across {trades_df['bucket'].nunique()} buckets")
    print(f"  Buckets   : {sorted(trades_df['bucket'].unique())}")

    # Save raw trades for reference
    os.makedirs("output", exist_ok=True)
    trades_df.to_csv("output/asset_analysis_trades.csv", index=False)

    # Run analysis
    stats = analyze(trades_df)
    report = format_report(stats)
    print(report)

    with open("output/asset_class_analysis.txt", "w") as f:
        f.write(report)

    print(f"\nReport saved to output/asset_class_analysis.txt")
    print(f"Completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
