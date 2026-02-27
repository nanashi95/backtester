"""
stress_test.py
──────────────
Shizuka Core v1 — Structural Stress Test Suite
Entry point.

Usage:
    python stress_test.py

Output:
    Prints full report to stdout.
    Writes Markdown report to output/stress/stress_report.md.

Runtime: ~15–25 minutes (24 engine runs + 1,000-path Monte Carlo).
"""

from __future__ import annotations

import os
import sys
import time

# ── Configure data loader before any engine imports ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data.loader as loader
import data.yf_data_loader as yf_loader
loader.configure(yf_loader.load_all_data, yf_loader.get_instrument_bucket)

# ── Stress suite imports (after loader is configured) ─────────────────────────
from stress.runner       import run_all
from stress.monte_carlo  import run_monte_carlo
from stress.report       import generate_report, classify

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "stress")


def main() -> None:
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Run all stress tests ──────────────────────────────────────────────────
    results = run_all()

    # ── Monte Carlo on baseline trades ────────────────────────────────────────
    print("\n[MC] Monte Carlo — 1,000 paths...", flush=True)
    t0 = time.time()
    mc = run_monte_carlo(
        trades_df      = results["baseline"]["trades"],
        initial_equity = 100_000.0,
        n_sims         = 1000,
    )
    results["monte_carlo"] = mc
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Generate Markdown report ──────────────────────────────────────────────
    print("\nAssembling report...", flush=True)
    report_md = generate_report(results)
    verdict   = classify(results)

    # ── Console output ────────────────────────────────────────────────────────
    sep = "═" * 72
    print(f"\n{sep}")
    print(report_md)
    print(sep)

    # ── Quick MC summary to console ───────────────────────────────────────────
    if "error" not in mc:
        print(f"\n  MC p5 CAGR  : {mc['cagr_p5']:+.2%}")
        print(f"  MC p95 MaxDD: {mc['max_dd_p95']:.2%}")
        print(f"  MC profitable: {mc['pct_profitable']:.1%}   ruin: {mc['pct_ruin']:.1%}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict_display = {
        "Fragile":            "FRAGILE",
        "Structurally Valid": "STRUCTURALLY VALID",
        "Strong":             "STRONG",
    }
    total_elapsed = time.time() - t_start
    print(f"\n{'─'*72}")
    print(f"  VERDICT: {verdict_display.get(verdict, verdict)}")
    print(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'─'*72}\n")

    # ── Write report to file ──────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "stress_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(f"Report written → {report_path}")


if __name__ == "__main__":
    main()
