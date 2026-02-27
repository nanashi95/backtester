"""
stress/report.py
────────────────
Markdown report template, red flag definitions, and classification logic
for the Shizuka Core v1 Structural Stress Test Suite.

Classification:
  Fragile            — any red flag triggered, OR fails Valid thresholds
  Structurally Valid — no red flags, meets Valid thresholds
  Strong             — no red flags, meets Strong thresholds
"""

from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Tuple


# ── Red flag catalogue ────────────────────────────────────────────────────────
# Each entry: (human description, code key)
# Code key is used by _check_red_flags to tag triggered flags.

RED_FLAG_CATALOGUE: List[Tuple[str, str]] = [
    ("MaxDD > 35% in any regime window",                           "regime_max_dd"),
    ("CAGR < 0% in any regime ≥ 1 year",                          "regime_negative_cagr"),
    ("Sharpe < 0 in any regime",                                   "regime_negative_sharpe"),
    ("Worst losing streak < −8R in any regime",                    "streak_loss"),
    ("MC p5 CAGR < −5%",                                          "mc_cagr_p5"),
    ("MC p95 MaxDD worse than −50%",                              "mc_max_dd_p95"),
    ("MC ruin rate > 5% (equity falls below 50% of start)",        "mc_ruin"),
    ("Single-sector removal changes CAGR by > 4pp",                "removal_sensitivity"),
    ("One sector > 60% of total R",                                "concentration_pnl"),
    ("Donchian period CAGR spread > 5pp (D40 vs D60)",            "param_donchian"),
    ("ATR trail multiple CAGR spread > 4pp (1.8× vs 2.2×)",       "param_atr_trail"),
    ("Risk scaling non-linearity residual > 2pp",                  "param_risk_nonlinear"),
    ("CAGR falls when cap increases from 4R → 6R or 6R → 8R",    "heat_nonmonotonic"),
]


# ── Classification thresholds ─────────────────────────────────────────────────

THRESHOLDS = {
    #                    Valid     Strong
    "baseline_cagr":   (0.05,     0.08),
    "baseline_sharpe": (0.25,     0.45),
    "baseline_mar":    (0.15,     0.30),
    "mc_p5_cagr":      (-0.05,    0.00),
    "worst_regime_dd": (-0.35,   -0.30),
}


# ── Red flag evaluation ───────────────────────────────────────────────────────

def _check_red_flags(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluate all red flag conditions.
    Returns list of triggered flag dicts: {'code', 'detail', 'value'}.
    """
    triggered: List[Dict[str, Any]] = []

    def flag(code: str, detail: str, value: Any = None) -> None:
        triggered.append({"code": code, "detail": detail, "value": value})

    # ── Regime flags ──────────────────────────────────────────────────────────
    for r in results.get("regime", []):
        label = r.get("label", "?")

        if r["max_dd"] < -0.35:
            flag("regime_max_dd",
                 f"{label}: MaxDD = {r['max_dd']:.2%}", r["max_dd"])

        if r.get("n_years", 0) >= 1.0 and r["cagr"] < 0.0:
            flag("regime_negative_cagr",
                 f"{label}: CAGR = {r['cagr']:.2%}", r["cagr"])

        sharpe = r.get("sharpe", float("nan"))
        if not math.isnan(sharpe) and sharpe < 0.0:
            flag("regime_negative_sharpe",
                 f"{label}: Sharpe = {sharpe:.2f}", sharpe)

        if r.get("worst_streak_r", 0.0) < -8.0:
            flag("streak_loss",
                 f"{label}: worst streak = {r['worst_streak_r']:.1f}R",
                 r["worst_streak_r"])

    # ── Monte Carlo flags ─────────────────────────────────────────────────────
    mc = results.get("monte_carlo", {})
    if mc and "error" not in mc:
        if mc.get("cagr_p5", 0.0) < -0.05:
            flag("mc_cagr_p5",
                 f"MC p5 CAGR = {mc['cagr_p5']:.2%}", mc["cagr_p5"])

        if mc.get("max_dd_p95", 0.0) < -0.50:
            flag("mc_max_dd_p95",
                 f"MC p95 MaxDD = {mc['max_dd_p95']:.2%}", mc["max_dd_p95"])

        if mc.get("pct_ruin", 0.0) > 0.05:
            flag("mc_ruin",
                 f"MC ruin rate = {mc['pct_ruin']:.1%}", mc["pct_ruin"])

    # ── Market removal flags ──────────────────────────────────────────────────
    for r in results.get("removal", []):
        delta = abs(r.get("cagr_delta", 0.0))
        if delta > 0.04:
            flag("removal_sensitivity",
                 f"Remove {r['removed_sector']}: ΔCAGR = {r['cagr_delta']:+.2%}",
                 r["cagr_delta"])

    # ── PnL concentration flag ─────────────────────────────────────────────────
    bucket_bd = results.get("baseline", {}).get("metrics", {}).get("bucket_breakdown", {})
    total_r = sum(b.get("total_r", 0.0) for b in bucket_bd.values())
    if total_r > 0:
        for bucket, b in bucket_bd.items():
            pct = b["total_r"] / total_r
            if pct > 0.60:
                flag("concentration_pnl",
                     f"{bucket} = {pct:.1%} of total R", pct)

    # ── Parameter stability flags ─────────────────────────────────────────────
    param = results.get("parameters", {})

    don_cagrs = [r["cagr"] for r in param.get("donchian", [])]
    if don_cagrs:
        spread = max(don_cagrs) - min(don_cagrs)
        if spread > 0.05:
            flag("param_donchian",
                 f"Donchian CAGR spread = {spread:.2%} (D40→D60)", spread)

    atr_cagrs = [r["cagr"] for r in param.get("atr_trail", [])]
    if atr_cagrs:
        spread = max(atr_cagrs) - min(atr_cagrs)
        if spread > 0.04:
            flag("param_atr_trail",
                 f"ATR trail CAGR spread = {spread:.2%} (1.8×→2.2×)", spread)

    risk_results = param.get("risk", [])
    if len(risk_results) == 3:
        from stress.runner import RISK_LEVELS
        levels = np.array(RISK_LEVELS)
        cagrs  = np.array([r["cagr"] for r in risk_results])
        coeffs = np.polyfit(levels, cagrs, 1)
        pred   = np.polyval(coeffs, levels)
        max_resid = float(np.abs(cagrs - pred).max())
        if max_resid > 0.02:
            flag("param_risk_nonlinear",
                 f"Risk scaling non-linearity = {max_resid:.2%}", max_resid)

    # ── Heat stress flag ──────────────────────────────────────────────────────
    heat = sorted(results.get("heat", []), key=lambda r: r["cap_level"])
    for i in range(len(heat) - 1):
        if heat[i + 1]["cagr"] < heat[i]["cagr"] - 0.01:
            flag("heat_nonmonotonic",
                 f"CAGR drops: cap {heat[i]['cap_level']:.0f}R "
                 f"({heat[i]['cagr']:.2%}) → "
                 f"{heat[i+1]['cap_level']:.0f}R "
                 f"({heat[i+1]['cagr']:.2%})")

    return triggered


# ── Classification ────────────────────────────────────────────────────────────

def classify(results: Dict[str, Any]) -> str:
    """
    Return 'Fragile', 'Structurally Valid', or 'Strong'.

    Logic:
      Any red flag  → Fragile
      No red flags + meets Valid thresholds  → Structurally Valid
      No red flags + meets Strong thresholds → Strong
      Otherwise → Fragile
    """
    flags = _check_red_flags(results)
    if flags:
        return "Fragile"

    bm  = results.get("baseline", {}).get("metrics", {})
    mc  = results.get("monte_carlo", {})
    reg = results.get("regime", [])

    worst_regime_dd = min((r["max_dd"] for r in reg), default=0.0)
    mc_p5_cagr      = mc.get("cagr_p5", 0.0) if mc and "error" not in mc else 0.0

    vals = {
        "baseline_cagr":   bm.get("cagr", 0.0),
        "baseline_sharpe": bm.get("sharpe", float("nan")),
        "baseline_mar":    bm.get("mar", 0.0),
        "mc_p5_cagr":      mc_p5_cagr,
        "worst_regime_dd": worst_regime_dd,
    }

    def meets(threshold_key: str, idx: int) -> bool:
        v = vals[threshold_key]
        t = THRESHOLDS[threshold_key][idx]
        if math.isnan(v):
            return False
        # For drawdown: less negative is better
        if threshold_key in ("worst_regime_dd",):
            return v >= t
        return v >= t

    all_keys = list(THRESHOLDS.keys())

    if all(meets(k, 1) for k in all_keys):
        return "Strong"

    if all(meets(k, 0) for k in all_keys):
        return "Structurally Valid"

    return "Fragile"


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(results: Dict[str, Any]) -> str:
    """
    Assemble the full Markdown stress test report.
    Sections match the 8 deliverables specified in the brief.
    """
    lines: List[str] = []

    def h2(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def h3(title: str) -> None:
        lines.append(f"\n### {title}\n")

    # ── Title ─────────────────────────────────────────────────────────────────
    lines.append("# Shizuka Core v1 — Structural Stress Test Report")
    lines.append("")
    lines.append("> **Objective**: Validate structural robustness. "
                 "No optimization. No rule changes. Diagnostics only.")

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    h2("1. Baseline — 2008–2025 (Production Settings)")

    bm = results.get("baseline", {}).get("metrics", {})
    lines += [
        "| Metric | Value |",
        "|--------|-------|",
        f"| CAGR | {bm.get('cagr', float('nan')):+.2%} |",
        f"| Max Drawdown | {bm.get('max_dd', float('nan')):.2%} |",
        f"| MAR | {bm.get('mar', float('nan')):.3f} |",
        f"| Sharpe | {bm.get('sharpe', float('nan')):.2f} |",
        f"| Profit Factor | {bm.get('profit_factor', float('nan')):.2f} |",
        f"| Total Trades | {bm.get('total_trades', 0)} |",
        f"| Trades / yr | {bm.get('trades_per_year_avg', 0):.0f} |",
        f"| Win Rate | {bm.get('win_rate', 0):.1%} |",
        f"| Avg Win (R) | {bm.get('avg_win_r', 0):+.2f}R |",
        f"| Avg Loss (R) | {bm.get('avg_loss_r', 0):+.2f}R |",
        f"| Expectancy | {bm.get('expectancy_r', 0):+.2f}R |",
        f"| Avg Hold (days) | {bm.get('avg_hold_days', 0):.1f}d |",
        f"| % Time in Drawdown | {bm.get('pct_time_in_dd', 0):.1%} |",
        f"| Worst Losing Streak | {bm.get('worst_streak_r', 0):.1f}R |",
        f"| Max Consec. Losses | {bm.get('max_consec_loss', 0)} |",
        f"| Longest Underwater | {bm.get('longest_uw_days', 0)} days |",
    ]

    # Bucket breakdown
    h3("1a. Sector PnL Breakdown")
    bucket_bd = bm.get("bucket_breakdown", {})
    total_r = sum(b.get("total_r", 0) for b in bucket_bd.values()) or 1.0
    lines += [
        "| Sector | Trades | Win% | Avg R | Total R | % of PnL |",
        "|--------|--------|------|-------|---------|----------|",
    ]
    for bucket, b in sorted(bucket_bd.items()):
        pct_pnl = b["total_r"] / total_r * 100
        lines.append(
            f"| {bucket} | {b['trade_count']} | {b['win_rate']:.0%} "
            f"| {b['avg_r']:+.2f}R | {b['total_r']:+.1f}R | {pct_pnl:+.1f}% |"
        )

    # ── 2. Regime stress tests ────────────────────────────────────────────────
    h2("2. Regime Stress Tests")
    lines += [
        "| Regime | CAGR | MaxDD | MAR | Sharpe | PF | Tr/yr | WR | AvgW | AvgL | Streak | %InDD | AvgHold |",
        "|--------|------|-------|-----|--------|-----|-------|-----|------|------|--------|-------|---------|",
    ]
    for r in results.get("regime", []):
        n_yrs = max(r.get("n_years", 1.0), 0.01)
        tpy   = f"{r['total_trades'] / n_yrs:.0f}"
        sharpe_str = f"{r['sharpe']:.2f}" if not math.isnan(r.get("sharpe", float("nan"))) else "n/a"
        lines.append(
            f"| {r.get('label', '?')} "
            f"| {r['cagr']:+.2%} "
            f"| {r['max_dd']:.2%} "
            f"| {r['mar']:.2f} "
            f"| {sharpe_str} "
            f"| {r['profit_factor']:.2f} "
            f"| {tpy} "
            f"| {r['win_rate']:.0%} "
            f"| {r['avg_win_r']:+.2f}R "
            f"| {r['avg_loss_r']:+.2f}R "
            f"| {r['worst_streak_r']:.1f}R "
            f"| {r['pct_time_in_dd']:.0%} "
            f"| {r['avg_hold_days']:.0f}d |"
        )

    # ── 3. Parameter stability ─────────────────────────────────────────────────
    h2("3. Parameter Stability (Full Period 2008–2025)")
    lines.append("_Grid tables only. Each dimension varied independently. "
                 "No optimization commentary._")

    param = results.get("parameters", {})

    h3("3a. Donchian Base Period")
    lines += [
        "| Period | CAGR | MaxDD | MAR | Sharpe | PF | Trades/yr |",
        "|--------|------|-------|-----|--------|-----|-----------|",
    ]
    for r in param.get("donchian", []):
        n_yrs = max(r.get("n_years", 1.0), 0.01)
        lines.append(
            f"| {r['param_label']} "
            f"| {r['cagr']:+.2%} "
            f"| {r['max_dd']:.2%} "
            f"| {r['mar']:.3f} "
            f"| {r['sharpe']:.2f} "
            f"| {r['profit_factor']:.2f} "
            f"| {r['total_trades'] / n_yrs:.0f} |"
        )

    h3("3b. ATR Trailing Stop Multiple")
    lines += [
        "| ATR Trail | CAGR | MaxDD | MAR | Sharpe | PF | Trades/yr |",
        "|-----------|------|-------|-----|--------|-----|-----------|",
    ]
    for r in param.get("atr_trail", []):
        n_yrs = max(r.get("n_years", 1.0), 0.01)
        lines.append(
            f"| {r['param_label']} "
            f"| {r['cagr']:+.2%} "
            f"| {r['max_dd']:.2%} "
            f"| {r['mar']:.3f} "
            f"| {r['sharpe']:.2f} "
            f"| {r['profit_factor']:.2f} "
            f"| {r['total_trades'] / n_yrs:.0f} |"
        )

    h3("3c. Risk Per Trade")
    lines += [
        "| Risk % | CAGR | MaxDD | MAR | Sharpe | PF | Trades/yr |",
        "|--------|------|-------|-----|--------|-----|-----------|",
    ]
    for r in param.get("risk", []):
        n_yrs = max(r.get("n_years", 1.0), 0.01)
        lines.append(
            f"| {r['param_label']} "
            f"| {r['cagr']:+.2%} "
            f"| {r['max_dd']:.2%} "
            f"| {r['mar']:.3f} "
            f"| {r['sharpe']:.2f} "
            f"| {r['profit_factor']:.2f} "
            f"| {r['total_trades'] / n_yrs:.0f} |"
        )

    # ── 4. Monte Carlo ─────────────────────────────────────────────────────────
    h2("4. Monte Carlo — 1,000 Trade Reshuffles")
    mc = results.get("monte_carlo", {})
    if not mc:
        lines.append("_Monte Carlo not run._")
    elif "error" in mc:
        lines.append(f"_Error: {mc['error']}_")
    else:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Simulations | {mc['n_simulations']:,} |",
            f"| Trades per path | {mc['n_trades_per_sim']:,} |",
            f"| Period modelled | {mc['n_years']:.1f} years |",
            f"| CAGR — p5 / p50 / p95 | {mc['cagr_p5']:+.2%} / {mc['cagr_p50']:+.2%} / {mc['cagr_p95']:+.2%} |",
            f"| MaxDD — p5 / p50 / p95 | {mc['max_dd_p5']:.2%} / {mc['max_dd_p50']:.2%} / {mc['max_dd_p95']:.2%} |",
            f"| % profitable paths | {mc['pct_profitable']:.1%} |",
            f"| % ruin paths (equity < 50%) | {mc['pct_ruin']:.1%} |",
        ]
        lines.append("")
        lines.append("_Method: bootstrap reshuffle of realized trade dollar PnL, "
                     "1,000 paths × full trade count. CAGR computed over actual "
                     "calendar period. No re-simulation of position sizing._")

    # ── 5. Market removal ──────────────────────────────────────────────────────
    h2("5. Market Removal — PnL Concentration and Dependency")
    lines += [
        "| Removed Sector | Instruments | CAGR | ΔCAGR | MaxDD | ΔMaxDD |",
        "|----------------|-------------|------|-------|-------|--------|",
    ]
    for r in results.get("removal", []):
        lines.append(
            f"| {r['removed_sector']} "
            f"| {r['removed_count']} "
            f"| {r['cagr']:+.2%} "
            f"| {r['cagr_delta']:+.2%} "
            f"| {r['max_dd']:.2%} "
            f"| {r['max_dd_delta']:+.2%} |"
        )

    # ── 6. Heat stress ─────────────────────────────────────────────────────────
    h2("6. Heat Stress — Portfolio Cap")
    lines += [
        "| Cap (R) | CAGR | MaxDD | MAR | Sharpe | PF |",
        "|---------|------|-------|-----|--------|-----|",
    ]
    for r in sorted(results.get("heat", []), key=lambda x: x["cap_level"]):
        lines.append(
            f"| {r['cap_level']:.0f}R "
            f"| {r['cagr']:+.2%} "
            f"| {r['max_dd']:.2%} "
            f"| {r['mar']:.3f} "
            f"| {r['sharpe']:.2f} "
            f"| {r['profit_factor']:.2f} |"
        )

    # ── 7. Red flag evaluation ─────────────────────────────────────────────────
    h2("7. Red Flag Evaluation")
    flags = _check_red_flags(results)

    if flags:
        lines.append(f"**{len(flags)} red flag(s) triggered:**\n")
        for f in flags:
            lines.append(f"- `[{f['code']}]` {f['detail']}")
    else:
        lines.append("**No red flags triggered.**")

    lines.append("")
    lines.append("**Complete red flag catalogue:**\n")
    lines += [
        "| Status | Code | Definition |",
        "|--------|------|------------|",
    ]
    triggered_codes = {f["code"] for f in flags}
    for desc, code in RED_FLAG_CATALOGUE:
        status = "FAIL" if code in triggered_codes else "pass"
        lines.append(f"| {status} | `{code}` | {desc} |")

    # ── 8. Classification ──────────────────────────────────────────────────────
    h2("8. Classification")
    verdict = classify(results)
    verdict_labels = {
        "Fragile":            "FRAGILE",
        "Structurally Valid": "STRUCTURALLY VALID",
        "Strong":             "STRONG",
    }
    lines.append(f"**Verdict: {verdict_labels.get(verdict, verdict)}**\n")

    # Threshold scorecard
    bm_vals = results.get("baseline", {}).get("metrics", {})
    mc_vals = results.get("monte_carlo", {}) or {}
    reg_dds = [r["max_dd"] for r in results.get("regime", [])]
    worst_dd = min(reg_dds) if reg_dds else float("nan")
    mc_p5    = mc_vals.get("cagr_p5", float("nan")) if "error" not in mc_vals else float("nan")

    lines += [
        "| Metric | Valid Threshold | Strong Threshold | Actual | Result |",
        "|--------|-----------------|------------------|--------|--------|",
    ]

    def _row(label: str, key: str, v: float, fmt: str, invert: bool = False) -> str:
        tv, ts = THRESHOLDS[key]
        actual_str = f"{v:{fmt}}" if not math.isnan(v) else "n/a"
        # invert=True means "less negative = better" (drawdown)
        ok_valid  = (v >= tv) if not invert else (v >= tv)
        ok_strong = (v >= ts) if not invert else (v >= ts)
        result = "Strong" if ok_strong else ("Valid" if ok_valid else "Fail")
        return (f"| {label} | {tv:{fmt}} | {ts:{fmt}} | {actual_str} | {result} |")

    lines.append(_row("Baseline CAGR",   "baseline_cagr",   bm_vals.get("cagr", float("nan")),   "+.2%"))
    lines.append(_row("Baseline Sharpe", "baseline_sharpe", bm_vals.get("sharpe", float("nan")), ".2f"))
    lines.append(_row("Baseline MAR",    "baseline_mar",    bm_vals.get("mar", float("nan")),    ".3f"))
    lines.append(_row("MC p5 CAGR",      "mc_p5_cagr",      mc_p5,                               "+.2%"))
    lines.append(_row("Worst regime DD", "worst_regime_dd", worst_dd,                             ".2%"))
    lines.append(f"| Red flags | 0 | 0 | {len(flags)} | {'Pass' if not flags else 'Fail'} |")

    lines.append("")
    lines.append("---")
    lines.append("_Report generated by `stress_test.py`. "
                 "Engine, rules, and risk parameters unchanged._")

    return "\n".join(lines)
