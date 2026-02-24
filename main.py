"""
main.py
-------
Entry point for the trend-following backtest engine.

Runs registered strategies in Mode A (ideal robot) and Mode B (human executable)
and prints a side-by-side comparison with Mode B cancellation/delay statistics.

Usage:
    python main.py

Outputs (per strategy × mode):
    - output/{slug}/equity_curve.csv
    - output/{slug}/trades.csv
    - output/{slug}/metrics_summary.txt

Comparison:
    - output/comparison.txt
"""

import sys
import os
import re
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from strategies import STRATEGIES
from engine.portfolio_engine import PortfolioEngine
from metrics.metrics_engine import compute_all_metrics, format_report


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def run_strategy(strategy, start, end, initial_equity, instruments=None, mode="A",
                 cap_per_bucket=1.0, cap_total=None, silent=False):
    """Run a single strategy in the given mode. Returns (metrics, equity_curve, trades_df, engine)."""
    cfg = strategy.config()
    if not silent:
        print(f"\n{'=' * 60}")
        print(f"  STRATEGY : {cfg.name}")
        print(f"  Mode     : {mode} ({'Ideal Robot' if mode == 'A' else 'Human Executable'})")
        print(f"  Period   : {start} -> {end}")
        print(f"  ATR({cfg.atr_period}) | Stop: {cfg.atr_initial_stop}x/{cfg.atr_trailing_stop}x")
        if instruments:
            print(f"  Universe : {len(instruments)} instruments")
        print(f"{'=' * 60}")

    engine = PortfolioEngine(
        strategy=strategy,
        start=start,
        end=end,
        initial_equity=initial_equity,
        instruments=instruments,
        mode=mode,
        cap_per_bucket=cap_per_bucket,
        cap_total=cap_total,
    )

    equity_curve = engine.run()
    trades_df    = engine.get_trades_df()

    if not silent:
        print("\nComputing performance metrics...")
    metrics = compute_all_metrics(equity_curve, trades_df, initial_equity=initial_equity)

    report = format_report(metrics)
    if not silent:
        print(report)

    # Save outputs (skipped in silent mode to avoid polluting output dirs)
    if silent:
        return metrics, equity_curve, trades_df, engine

    period_tag = f"{start[:4]}{start[5:7]}_{end[:4]}{end[5:7]}"
    slug = f"{_slug(cfg.name)}_mode{mode}_{period_tag}"
    out_dir = os.path.join("output", slug)
    os.makedirs(out_dir, exist_ok=True)

    equity_curve.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    with open(os.path.join(out_dir, "metrics_summary.txt"), "w") as f:
        f.write(report)
        f.write("\n\nRaw metrics:\n")
        for k, v in metrics.items():
            if k not in ("underwater_periods", "r_distribution",
                          "bucket_breakdown", "exposure_clustering",
                          "annual_returns", "trades_per_year"):
                f.write(f"  {k}: {v}\n")

    return metrics, equity_curve, trades_df, engine


def format_comparison(all_results):
    """Side-by-side comparison table."""
    lines = []
    lines.append("\n" + "=" * 118)
    lines.append("  STRATEGY COMPARISON  (Mode A = Ideal Robot | Mode B = Human Executable)")
    lines.append("=" * 118)

    header = (f"  {'Label':<34s} {'CAGR':>7s} {'MaxDD':>7s} {'MAR':>6s} {'Sharpe':>7s} "
              f"{'Trades':>6s} {'WinRate':>7s} {'Expect':>7s} {'PF':>6s} "
              f"{'AvgHold':>7s}")
    lines.append(header)
    lines.append("  " + "-" * 114)

    for label, m in all_results:
        cagr     = m.get("cagr", 0) * 100
        max_dd   = m.get("max_drawdown", 0) * 100
        mar      = m.get("mar_ratio", 0)
        sharpe   = m.get("sharpe", float("nan"))
        trades   = m.get("total_trades", 0)
        win_rate = m.get("win_rate", 0) * 100
        expect   = m.get("expectancy_r", 0)
        pf       = m.get("profit_factor", 0)
        avg_hold = m.get("avg_hold_days", 0)

        sharpe_s = f"{sharpe:>6.2f}" if sharpe == sharpe else "   nan"
        row = (f"  {label:<34s} {cagr:>6.2f}% {max_dd:>6.2f}% {mar:>6.2f} {sharpe_s} "
               f"{trades:>6d} {win_rate:>6.1f}% {expect:>6.2f}R {pf:>6.2f} "
               f"{avg_hold:>6.1f}d")
        lines.append(row)

    lines.append("=" * 118)
    return "\n".join(lines)


def print_mode_b_stats(engine: PortfolioEngine, label: str) -> None:
    """Print Mode B cancellation and delay statistics."""
    stats = engine.get_mode_b_stats()
    total_cancelled = stats["cancelled_sleep_window"] + stats["cancelled_too_late"]

    print(f"\n{'=' * 60}")
    print(f"  MODE B STATS  — {label}")
    print(f"{'=' * 60}")
    print(f"  Cancelled (sleep window / 24h timeout) : {stats['cancelled_sleep_window']}")
    print(f"  Cancelled (too late / ATR-distance gate): {stats['cancelled_too_late']}")
    print(f"  Total cancelled                         : {total_cancelled}")

    dd = stats["delay_distribution"]
    if dd:
        total_entered = sum(dd.values())
        print(f"\n  Entry delay vs Mode A (H4 bars, {total_entered} trades entered):")
        for bars in sorted(dd):
            count = dd[bars]
            pct   = count / total_entered * 100
            note  = "  ← same speed as Mode A" if bars == 0 else ""
            print(f"    {bars:2d} bar(s) delayed : {count:4d}  ({pct:5.1f}%){note}")
    else:
        print("\n  No trades entered in Mode B.")

    samples = stats["cancelled_samples"]
    if samples:
        print(f"\n  Sample cancelled trades ({len(samples)}):")
        for s in samples:
            ref    = f"{s['ref_entry_price']:.5f}" if s["ref_entry_price"] is not None else "N/A"
            actual = f"{s['actual_entry_price']:.5f}"
            dist   = f"{s['distance']:.5f}" if s["distance"] is not None else "N/A"
            print(f"    [{s['reason']:12s}] {s['instrument']:<8s} "
                  f"sig={s['signal_ts']}  ref={ref}  actual={actual}  "
                  f"atr={s['atr_signal']:.5f}  dist={dist}")
    print("=" * 60)


def main():
    t0 = time.time()

    initial_equity = 100_000.0

    # ── Configuration ─────────────────────────────────────────────────────────
    # 5-bucket diversified config: equity + FX + energy + metals + agriculture
    instruments = [
        # Equity (6)
        "US100", "US500", "US2000", "DE30", "JP225", "GB100",
        # FX (6)
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "USDNOK",
        # Energy (3)
        "USOil", "UKOil", "NATGAS",
        # Metals (3)
        "Gold", "Silver", "Copper",
        # Agriculture (3)
        "WHEAT", "SOYBEAN", "Sugar",
    ]

    periods = [
        ("2008-01-01", "2016-12-31"),
        ("2017-01-01", "2022-12-31"),
        ("2023-01-01", "2025-12-31"),
        ("2008-01-01", "2025-12-31"),  # full period
    ]

    modes = ["A"]
    # ─────────────────────────────────────────────────────────────────────────

    n_inst = str(len(instruments)) if instruments else "all"
    print("=" * 60)
    print("  TREND FOLLOWING BACKTEST ENGINE")
    print(f"  Strategies : {len(STRATEGIES)} registered")
    print(f"  Universe   : {n_inst} instruments")
    print(f"  Periods    : {', '.join(f'{s} -> {e}' for s, e in periods)}")
    print(f"  Modes      : {', '.join(modes)}")
    from engine.risk_engine import RISK_PER_TRADE
    print(f"  Risk       : {RISK_PER_TRADE*100:.1f}% per trade")
    print("=" * 60)

    all_results   = []   # (label, metrics) for comparison table
    mode_b_engines = []  # (label, engine) for Mode B stats

    for strategy in STRATEGIES:
        cfg = strategy.config()
        for start, end in periods:
            for mode in modes:
                label = f"{cfg.name} [M{mode}] {start[:7]}→{end[:7]}"
                metrics, equity_curve, trades_df, engine = run_strategy(
                    strategy, start, end, initial_equity,
                    instruments=instruments, mode=mode,
                )
                all_results.append((label, metrics))
                if mode == "B":
                    mode_b_engines.append((label, engine))

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison = format_comparison(all_results)
    print(comparison)

    os.makedirs("output", exist_ok=True)
    with open("output/comparison.txt", "w") as f:
        f.write(comparison)

    # ── Mode B statistics ─────────────────────────────────────────────────────
    for label, engine in mode_b_engines:
        print_mode_b_stats(engine, label)

    elapsed = time.time() - t0
    print(f"\nAll backtests complete in {elapsed:.1f}s")
    print("Output files saved to ./output/")

    return all_results


if __name__ == "__main__":
    main()
