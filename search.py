"""
search.py
---------
Systematic search over bucket combinations and risk cap parameters.

Searches:
  Phase 1 — bucket selection:
    Tests every meaningful subset of the 5 new-universe buckets
    (equity, fx_yen, fx_maj, energy, metals, softs) with 1R/bucket cap.
    FX is split back into yen and majors for testing — both share the "fx"
    bucket slot, so they compete correctly when mixed.

  Phase 2 — cap variation:
    For the top N configs by 2008-2025 MAR, re-runs with cap_per_bucket=2R
    (allows 2 concurrent trades per bucket).

Usage:
    python3 search.py

Output:
    output/search_results.txt   — full sorted comparison table
    Prints progress and final table to stdout.
"""

from __future__ import annotations
import sys, os, time, io
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import redirect_stdout, redirect_stderr
from strategies import STRATEGIES
from main import run_strategy

# ── Instrument groups ─────────────────────────────────────────────────────────
METALS  = ["Gold", "Silver", "Copper"]
EQUITY  = ["US100", "US500", "US2000", "DE30", "JP225", "GB100"]
FX_YEN  = ["AUDJPY", "EURJPY", "USDJPY", "CADJPY", "NZDJPY", "CHFJPY"]
FX_MAJ  = ["EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "USDNOK"]
FX_ALL  = FX_YEN + FX_MAJ
ENERGY  = ["USOil", "UKOil", "NATGAS"]
SOFTS   = ["WHEAT", "SOYBEAN", "Sugar"]


def _buckets_of(instruments: list[str]) -> int:
    """Count distinct active buckets in the instrument list."""
    from engine.risk_engine import BUCKET_MAP
    return len({BUCKET_MAP[i] for i in instruments if i in BUCKET_MAP})


# ── Phase 1: bucket selection configs (1R/bucket) ────────────────────────────
BUCKET_CONFIGS: list[tuple[str, list[str]]] = [
    # Single buckets
    ("metals_only",          METALS),
    ("equity_only",          EQUITY),
    ("fx_yen_only",          FX_YEN),
    ("fx_maj_only",          FX_MAJ),
    ("energy_only",          ENERGY),
    ("softs_only",           SOFTS),
    # Two-bucket combos with metals (best single-bucket candidate)
    ("metals+equity",        METALS + EQUITY),
    ("metals+fx_yen",        METALS + FX_YEN),
    ("metals+fx_maj",        METALS + FX_MAJ),
    ("metals+fx_all",        METALS + FX_ALL),
    ("metals+energy",        METALS + ENERGY),
    ("metals+softs",         METALS + SOFTS),
    # Three-bucket combos
    ("metals+equity+fx_yen", METALS + EQUITY + FX_YEN),
    ("metals+equity+fx_all", METALS + EQUITY + FX_ALL),
    ("metals+equity+energy", METALS + EQUITY + ENERGY),
    ("metals+equity+softs",  METALS + EQUITY + SOFTS),
    ("metals+fx_yen+energy", METALS + FX_YEN + ENERGY),
    ("metals+fx_yen+softs",  METALS + FX_YEN + SOFTS),
    # Four-bucket combos (curated — skip majors-heavy versions)
    ("metals+eq+fxyen+energy",    METALS + EQUITY + FX_YEN + ENERGY),
    ("metals+eq+fxyen+softs",     METALS + EQUITY + FX_YEN + SOFTS),
    ("metals+eq+energy+softs",    METALS + EQUITY + ENERGY + SOFTS),
    ("metals+fxyen+energy+softs", METALS + FX_YEN + ENERGY + SOFTS),
    # Full new universe (reference)
    ("all_new_universe",     METALS + EQUITY + FX_ALL + ENERGY + SOFTS),
]

PERIODS = [
    ("2008-01-01", "2016-12-31"),
    ("2017-01-01", "2022-12-31"),
    ("2023-01-01", "2025-12-31"),
    ("2008-01-01", "2025-12-31"),
]
FULL_PERIOD = ("2008-01-01", "2025-12-31")

INITIAL_EQUITY = 100_000.0


def run_silent(strategy, start, end, instruments, cap_per_bucket=1.0, cap_total=None):
    """Run backtest capturing all prints; return metrics dict."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        metrics, *_ = run_strategy(
            strategy, start, end, INITIAL_EQUITY,
            instruments=instruments,
            mode="A",
            cap_per_bucket=cap_per_bucket,
            cap_total=cap_total,
            silent=True,
        )
    return metrics


def fmt_row(label, m_full, period_mars):
    """Format a comparison table row."""
    cagr    = m_full.get("cagr", 0) * 100
    max_dd  = m_full.get("max_drawdown", 0) * 100
    mar     = m_full.get("mar_ratio", 0)
    sharpe  = m_full.get("sharpe", float("nan"))
    trades  = m_full.get("total_trades", 0)
    pf      = m_full.get("profit_factor", 0)
    sharpe_s = f"{sharpe:5.2f}" if sharpe == sharpe else "  nan"
    # Period MAR columns
    mar_cols = "  ".join(f"{v:5.2f}" for v in period_mars)
    return (f"  {label:<30s} {cagr:>6.2f}% {max_dd:>7.2f}% {mar:>5.2f}  {sharpe_s}  "
            f"{trades:>5d}  {pf:>5.2f}  {mar_cols}")


def search_phase1(strategy) -> list[tuple[str, dict, list[float]]]:
    """Run all bucket configs for all periods. Returns list of (label, full_metrics, period_mars)."""
    results = []
    total = len(BUCKET_CONFIGS)
    print(f"\nPhase 1: {total} bucket configs × {len(PERIODS)} periods = {total * len(PERIODS)} backtests")

    for i, (name, instruments) in enumerate(BUCKET_CONFIGS):
        n_b = _buckets_of(instruments)
        cap_total = n_b * 1.0  # 1R per bucket, total = n_buckets
        label = f"{name} [1R/b, cap={cap_total:.0f}R]"

        period_metrics = {}
        for start, end in PERIODS:
            tag = f"{start[:4]}–{end[:4]}"
            m = run_silent(strategy, start, end, instruments,
                           cap_per_bucket=1.0, cap_total=cap_total)
            period_metrics[(start, end)] = m

        full_m = period_metrics[FULL_PERIOD]
        period_mars = [
            period_metrics[p].get("mar_ratio", 0)
            for p in PERIODS[:3]  # first 3 periods (excl. full)
        ]
        results.append((label, full_m, period_mars))

        mar = full_m.get("mar_ratio", 0)
        cagr = full_m.get("cagr", 0) * 100
        print(f"  [{i+1:2d}/{total}] {name:<30s}  CAGR={cagr:+5.2f}%  MAR={mar:.3f}")

    return results


def search_phase2(strategy, top_configs: list[tuple[str, list[str]]], cap_per_bucket=2.0
                  ) -> list[tuple[str, dict, list[float]]]:
    """Re-run top configs with higher cap_per_bucket."""
    results = []
    print(f"\nPhase 2: {len(top_configs)} configs with {cap_per_bucket}R/bucket")

    for name, instruments in top_configs:
        n_b = _buckets_of(instruments)
        cap_total = n_b * cap_per_bucket
        label = f"{name} [{cap_per_bucket:.0f}R/b, cap={cap_total:.0f}R]"

        period_metrics = {}
        for start, end in PERIODS:
            m = run_silent(strategy, start, end, instruments,
                           cap_per_bucket=cap_per_bucket, cap_total=cap_total)
            period_metrics[(start, end)] = m

        full_m = period_metrics[FULL_PERIOD]
        period_mars = [
            period_metrics[p].get("mar_ratio", 0)
            for p in PERIODS[:3]
        ]
        results.append((label, full_m, period_mars))

        mar = full_m.get("mar_ratio", 0)
        cagr = full_m.get("cagr", 0) * 100
        print(f"  {name:<30s}  CAGR={cagr:+5.2f}%  MAR={mar:.3f}")

    return results


def print_table(all_results: list[tuple[str, dict, list[float]]], title: str) -> str:
    """Print and return a sorted comparison table."""
    # Sort by 2008-2025 MAR descending
    all_results.sort(key=lambda x: x[1].get("mar_ratio", -999), reverse=True)

    header = (f"\n{'=' * 110}\n"
              f"  {title}\n"
              f"{'=' * 110}\n"
              f"  {'Config':<30s} {'CAGR':>7s} {'MaxDD':>8s} {'MAR':>6s} {'Sharpe':>6s} "
              f"{'Trd':>5s} {'PF':>5s}  "
              f"{'08-16MAR':>8s}  {'17-22MAR':>8s}  {'23-25MAR':>8s}\n"
              f"  {'-' * 106}")
    lines = [header]
    for label, m, pmars in all_results:
        lines.append(fmt_row(label, m, pmars))
    lines.append("=" * 110)
    table = "\n".join(lines)
    print(table)
    return table


def main():
    t0 = time.time()
    strategy = STRATEGIES[0]  # DonchianReversal

    print("=" * 60)
    print("  BUCKET + RISK PARAMETER SEARCH")
    print(f"  Strategy : {strategy.config().name}")
    print(f"  Periods  : {len(PERIODS)} periods per config")
    print("=" * 60)

    # ── Phase 1: bucket selection ─────────────────────────────────────────────
    phase1 = search_phase1(strategy)

    table1 = print_table(list(phase1), "PHASE 1 — BUCKET SELECTION  (1R per bucket, sorted by 2008-2025 MAR)")

    # Pick top 5 configs by full-period MAR for phase 2
    phase1_sorted = sorted(phase1, key=lambda x: x[1].get("mar_ratio", -999), reverse=True)
    top5_names = [(r[0].split(" [")[0], None) for r in phase1_sorted[:5]]
    top5_instruments = []
    bucket_cfg_map = {name: instr for name, instr in BUCKET_CONFIGS}
    for name, _ in top5_names:
        if name in bucket_cfg_map:
            top5_instruments.append((name, bucket_cfg_map[name]))

    # ── Phase 2: cap_per_bucket = 2R on top 5 ────────────────────────────────
    phase2 = search_phase2(strategy, top5_instruments, cap_per_bucket=2.0)

    # Combined view: top-10 from phase1 + all of phase2
    combined = phase1_sorted[:10] + phase2
    table2 = print_table(combined, "COMBINED — Top Phase-1 configs + Phase-2 (2R/bucket) variants")

    # Save
    os.makedirs("output", exist_ok=True)
    with open("output/search_results.txt", "w") as f:
        f.write(table1 + "\n\n" + table2)
    print(f"\nResults saved to output/search_results.txt")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
