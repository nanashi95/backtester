"""
main.py
-------
Entry point for the trend-following backtest engine.

Runs all registered strategies and prints a side-by-side comparison table.

Usage:
    python main.py

Outputs (per strategy):
    - output/{strategy_name}/equity_curve.csv
    - output/{strategy_name}/trades.csv
    - output/{strategy_name}/metrics_summary.txt

Comparison:
    - output/comparison.txt
"""

import sys
import os
import re
import time
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from strategies import STRATEGIES
from engine.portfolio_engine import PortfolioEngine
from metrics.metrics_engine import compute_all_metrics, format_report


def _slug(name: str) -> str:
    """Convert strategy display name to filesystem-safe slug."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def run_strategy(strategy, start, end, initial_equity, instruments=None):
    """Run a single strategy and return (metrics, equity_curve, trades_df)."""
    cfg = strategy.config()
    print(f"\n{'=' * 60}")
    print(f"  STRATEGY: {cfg.name}")
    print(f"  Period: {start} -> {end}")
    print(f"  ATR({cfg.atr_period}) | Stop: {cfg.atr_initial_stop}x/{cfg.atr_trailing_stop}x"
          f" | Reversal exit: {'ON' if cfg.use_reversal_exit else 'OFF'}")
    if instruments:
        print(f"  Universe: {len(instruments)} instruments")
    print(f"{'=' * 60}")

    engine = PortfolioEngine(
        strategy=strategy,
        start=start,
        end=end,
        initial_equity=initial_equity,
        instruments=instruments,
    )

    equity_curve = engine.run()
    trades_df    = engine.get_trades_df()

    print("\nComputing performance metrics...")
    metrics = compute_all_metrics(equity_curve, trades_df,
                                  initial_equity=initial_equity)

    report = format_report(metrics)
    print(report)

    # Save per-strategy outputs
    period_tag = f"{start[:4]}_{end[:4]}"
    slug = f"{_slug(cfg.name)}_{period_tag}"
    out_dir = os.path.join("output", slug)
    os.makedirs(out_dir, exist_ok=True)

    equity_curve.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    with open(os.path.join(out_dir, "metrics_summary.txt"), "w") as f:
        f.write(report)
        f.write("\n\nRaw metrics dict:\n")
        for k, v in metrics.items():
            if k not in ("underwater_periods", "r_distribution",
                          "bucket_breakdown", "exposure_clustering",
                          "annual_returns", "trades_per_year"):
                f.write(f"  {k}: {v}\n")

    return metrics, equity_curve, trades_df


def format_comparison(all_results):
    """Build a side-by-side comparison table from collected results."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("  STRATEGY COMPARISON")
    lines.append("=" * 100)

    # Header
    header = (f"  {'Strategy':<28s} {'CAGR':>7s} {'MaxDD':>7s} {'MAR':>6s} "
              f"{'Trades':>6s} {'WinRate':>7s} {'Expect':>7s} {'PF':>6s} "
              f"{'AvgHold':>7s}")
    lines.append(header)
    lines.append("  " + "-" * 96)

    for name, m in all_results:
        cagr     = m.get("cagr", 0) * 100
        max_dd   = m.get("max_drawdown", 0) * 100
        mar      = m.get("mar_ratio", 0)
        trades   = m.get("total_trades", 0)
        win_rate = m.get("win_rate", 0) * 100
        expect   = m.get("expectancy_r", 0)
        pf       = m.get("profit_factor", 0)
        avg_hold = m.get("avg_hold_days", 0)

        row = (f"  {name:<28s} {cagr:>6.2f}% {max_dd:>6.2f}% {mar:>6.2f} "
               f"{trades:>6d} {win_rate:>6.1f}% {expect:>6.2f}R {pf:>6.2f} "
               f"{avg_hold:>6.1f}d")
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    t0 = time.time()

    initial_equity = 100_000.0

    # Full universe
    instruments = None  # None = all 16 instruments

    # Periods
    periods = [
        ("2025-01-01", "2025-07-31"),
        ("2025-08-01", "2025-12-31"),
        ("2025-01-01", "2025-12-31"),
    ]

    n_inst = str(len(instruments)) if instruments else "all 16"
    print("=" * 60)
    print("  TREND FOLLOWING BACKTEST ENGINE")
    print(f"  Strategies: {len(STRATEGIES)} registered")
    print(f"  Universe: {n_inst} instruments")
    print(f"  Periods: {len(periods)} ({', '.join(f'{s[:4]}-{e[:4]}' for s,e in periods)})")
    print(f"  Risk: 0.7% per trade")
    print("=" * 60)

    all_results = []

    for strategy in STRATEGIES:
        cfg = strategy.config()
        for start, end in periods:
            label = f"{cfg.name} [{start[:4]}-{end[:4]}]"
            metrics, equity_curve, trades_df = run_strategy(
                strategy, start, end, initial_equity, instruments=instruments
            )
            all_results.append((label, metrics))

    # ── Comparison ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        comparison = format_comparison(all_results)
        print(comparison)
    elif len(all_results) == 1:
        comparison = format_comparison(all_results)
        print(comparison)

    # Save comparison
    os.makedirs("output", exist_ok=True)
    comparison_text = format_comparison(all_results)
    with open("output/comparison.txt", "w") as f:
        f.write(comparison_text)

    elapsed = time.time() - t0
    print(f"\nAll backtests complete in {elapsed:.1f}s")
    print("Output files saved to ./output/")

    if len(all_results) == 1:
        name, metrics = all_results[0]
        slug = _slug(name)
        print(f"  {slug}/equity_curve.csv")
        print(f"  {slug}/trades.csv")
        print(f"  {slug}/metrics_summary.txt")
    else:
        for name, _ in all_results:
            slug = _slug(name)
            print(f"  {slug}/")
    print("  comparison.txt")

    return all_results


if __name__ == "__main__":
    main()
