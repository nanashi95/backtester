"""
rates_fix_backtest.py
---------------------
Test rates sleeve period change: D150/D300 → D50/D100.
All other settings identical to speed-diff baseline.

Baseline (speed-diff, rates D150/D300):
  CAGR +9.69%  MaxDD -30.98%  MAR 0.313  Sharpe 0.50  Longest UW 2010d
  Rates PnL: -$21k (-3.4% of portfolio)
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import data.loader as loader
import data.yf_data_loader as yf_loader
loader.configure(yf_loader.load_all_data, yf_loader.get_instrument_bucket)

from strategies.donchian_ensemble import make_sleeve
from engine.ensemble_engine import EnsemblePortfolioEngine
from metrics.metrics_engine import compute_all_metrics

# ── Universe ──────────────────────────────────────────────────────────────────

RATES         = ["ZN", "ZB", "ZF"]
EQUITY        = ["ES", "NQ", "RTY", "YM"]
METALS        = ["GC", "SI", "HG", "PA", "PL"]
ENERGY        = ["CL", "NG"]
AGRICULTURE   = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

METALS_ENERGY = METALS + ENERGY
EQUITY_AGRI   = EQUITY + AGRICULTURE
ALL_NO_FX     = RATES + EQUITY + METALS + ENERGY + AGRICULTURE

BUCKET_ORDER  = ["rates", "equity", "metals", "energy", "agriculture"]
BUCKET_LABELS = {"rates": "Rates", "equity": "Equity", "metals": "Metals",
                 "energy": "Energy", "agriculture": "Agriculture"}

ASSET_CLASS_MAP: dict = {}
for _i in RATES:       ASSET_CLASS_MAP[_i] = "rates"
for _i in EQUITY:      ASSET_CLASS_MAP[_i] = "equity"
for _i in METALS:      ASSET_CLASS_MAP[_i] = "metals"
for _i in ENERGY:      ASSET_CLASS_MAP[_i] = "energy"
for _i in AGRICULTURE: ASSET_CLASS_MAP[_i] = "agriculture"

# ── Risk ──────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0
START, END     = "2008-01-01", "2025-12-31"
W = 72

BASELINE = {
    "cagr": 0.0969, "max_drawdown": -0.3098,
    "mar_ratio": 0.313, "sharpe": 0.50,
    "longest_underwater_days": 2010,
    "rates_pnl": -21724.0,
}

PERIODS = [
    ("2008-01-01", "2016-12-31", "2008–2016"),
    ("2017-01-01", "2022-12-31", "2017–2022"),
    ("2023-01-01", "2025-12-31", "2023–2025"),
    ("2008-01-01", "2025-12-31", "2008–2025 (Full)"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v == v else "nan"

def ann_vol_pct(eq: pd.DataFrame) -> float:
    s = eq["total_value"].resample("D").last().ffill()
    dr = s.pct_change().dropna()
    return float(dr.std() * np.sqrt(252) * 100) if len(dr) > 1 else float("nan")

def make_strategies_fixed():
    """Rates changed to D50/D100; everything else unchanged."""
    return {
        "B_ME": make_sleeve(50,  "B"),
        "C_ME": make_sleeve(100, "C"),
        "B_EA": make_sleeve(75,  "B"),
        "C_EA": make_sleeve(150, "C"),
        "B_R":  make_sleeve(50,  "B"),   # was 150
        "C_R":  make_sleeve(100, "C"),   # was 300
    }

SLEEVE_INSTRUMENTS = {
    "B_ME": METALS_ENERGY, "C_ME": METALS_ENERGY,
    "B_EA": EQUITY_AGRI,   "C_EA": EQUITY_AGRI,
    "B_R":  RATES,         "C_R":  RATES,
}

def run(start, end):
    eng = EnsemblePortfolioEngine(
        strategies         = make_strategies_fixed(),
        start              = start,
        end                = end,
        initial_equity     = INITIAL_EQUITY,
        instruments        = ALL_NO_FX,
        risk_per_trade     = RISK_PER_TRADE,
        global_cap         = GLOBAL_CAP,
        sleeve_instruments = SLEEVE_INSTRUMENTS,
        use_vol_scaling    = False,
    )
    eq     = eng.run()
    trades = eng.get_trades_df()
    return eq, trades

# ── Printers ──────────────────────────────────────────────────────────────────

def print_metrics(m, eq, label):
    n_yr = m.get("n_years", 1)
    tr   = m.get("total_trades", 0)
    print(f"\n{'═'*W}")
    print(f"  PORTFOLIO METRICS — {label}")
    print(f"{'═'*W}")
    print(f"  CAGR              : {m['cagr']*100:+.2f}%")
    print(f"  Max Drawdown      : {m['max_drawdown']*100:.2f}%")
    print(f"  MAR               : {fmt_f(m.get('mar_ratio', float('nan')), 3)}")
    print(f"  Sharpe            : {fmt_f(m.get('sharpe', float('nan')), 2)}")
    print(f"  Profit Factor     : {fmt_f(m.get('profit_factor', float('nan')), 2)}")
    print(f"  Longest UW        : {m.get('longest_underwater_days', 0)} days")
    print(f"  Ann. Volatility   : {fmt_f(ann_vol_pct(eq), 2)}%")
    print(f"  Total Trades      : {tr}  ({tr/n_yr:.0f}/yr)")
    print(f"  Win Rate          : {m.get('win_rate', 0)*100:.1f}%")
    print(f"  Expectancy        : {fmt_f(m.get('expectancy_r', 0), 3)}R")

def print_comparison(m):
    print(f"\n{'─'*W}")
    print(f"  DELTA vs SPEED-DIFF BASELINE (Rates D150/D300)")
    print(f"{'─'*W}")
    print(f"  {'Metric':<22s} {'New':>10s}  {'Baseline':>10s}  {'Delta':>10s}")
    print(f"  {'─'*58}")
    for label, key, scale, unit in [
        ("CAGR",       "cagr",                    100, "%"),
        ("MaxDD",      "max_drawdown",             100, "%"),
        ("MAR",        "mar_ratio",                  1, ""),
        ("Sharpe",     "sharpe",                     1, ""),
        ("Longest UW", "longest_underwater_days",    1, "d"),
    ]:
        nv = m.get(key, float("nan"))
        bv = BASELINE.get(key, float("nan"))
        if nv == nv and bv == bv:
            d = (nv - bv) * scale
            print(f"  {label:<22s} {nv*scale:>9.2f}{unit}  "
                  f"{bv*scale:>9.2f}{unit}  {'+' if d>=0 else ''}{d:.2f}{unit}")
        else:
            print(f"  {label:<22s} {'nan':>10s}  {'nan':>10s}  {'nan':>10s}")

def print_asset_class(trades, label=""):
    if label:
        print(f"\n{'─'*W}")
        print(f"  ASSET CLASS — {label}")
        print(f"{'─'*W}")
    total_pnl = trades["pnl"].sum()
    print(f"  {'Class':<14s} {'Trades':>6s} {'Expect':>8s} {'AvgHold':>8s} "
          f"{'Win%':>6s} {'TotalPnL':>12s}")
    print(f"  {'─'*62}")
    for bkt in BUCKET_ORDER:
        seg = trades[trades["bucket"] == bkt]
        lbl = BUCKET_LABELS[bkt]
        if seg.empty:
            print(f"  {lbl:<14s}  — no trades —")
            continue
        r    = seg["r_multiple"].dropna().values
        pnl  = seg["pnl"].sum()
        pct  = pnl / total_pnl * 100 if total_pnl else 0.0
        exp  = float(np.mean(r))       if len(r) else float("nan")
        hold = float(seg["hold_days"].mean()) if "hold_days" in seg.columns else float("nan")
        wr   = 100.0 * float(np.mean(r > 0)) if len(r) else float("nan")
        baseline_pnl = BASELINE.get("rates_pnl", 0.0) if bkt == "rates" else None
        delta_str = (f"  Δ{pnl - baseline_pnl:+.0f}$"
                     if baseline_pnl is not None else "")
        print(f"  {lbl:<14s}"
              f" {len(seg):>6d}"
              f" {exp:>+7.3f}R"
              f"  {hold:>6.1f}d"
              f" {wr:>5.1f}%"
              f" {pnl:>+11.0f}$  ({pct:>+5.1f}%){delta_str}")

def print_regime_table(period_results):
    print(f"\n{'─'*W}")
    print(f"  REGIME TABLE — Rates D50/D100")
    print(f"{'─'*W}")
    print(f"  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'MAR':>6s}  "
          f"{'Sharpe':>6s}  {'Trades/yr':>10s}")
    print(f"  {'─'*62}")
    for label, m in period_results:
        n_yr = m.get("n_years", 1)
        tr   = m.get("total_trades", 0)
        print(f"  {label:<16s}  "
              f"{m['cagr']*100:>+6.2f}%  "
              f"{m['max_drawdown']*100:>6.2f}%  "
              f"{fmt_f(m.get('mar_ratio', float('nan')), 3):>6s}  "
              f"{fmt_f(m.get('sharpe', float('nan')), 2):>6s}  "
              f"{tr/n_yr:>8.0f}/yr")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * W)
    print("  RATES FIX BACKTEST — Rates D50/D100  (was D150/D300)")
    print(f"  Metals/Energy : B=D50   C=D100  (unchanged)")
    print(f"  Equity/Agri   : B=D75   C=D150  (unchanged)")
    print(f"  Rates         : B=D50   C=D100  (changed from D150/D300)")
    print(f"  No FX  |  No vol-scaling  |  No sector caps")
    print("=" * W)

    # Full period
    eq, trades = run(START, END)
    metrics = compute_all_metrics(eq, trades, INITIAL_EQUITY)

    print_metrics(metrics, eq, "2008–2025  Rates D50/D100")
    print_comparison(metrics)
    print_asset_class(trades, "2008–2025")

    # Per-regime breakdown
    period_results = []
    for start, end, label in PERIODS:
        print(f"\n  Running {label}...")
        eq_p, tr_p = run(start, end)
        m_p = compute_all_metrics(eq_p, tr_p, INITIAL_EQUITY)
        period_results.append((label, m_p))

    print_regime_table(period_results)

    # Rates-specific detail
    print(f"\n{'─'*W}")
    print(f"  RATES INSTRUMENTS DETAIL — 2008–2025")
    print(f"{'─'*W}")
    eq_full, trades_full = run(START, END)
    r_trades = trades_full[trades_full["bucket"] == "rates"]
    print(f"  {'Instrument':<10s} {'Trades':>7s} {'Expect':>8s} "
          f"{'Win%':>7s} {'TotalPnL':>12s}")
    print(f"  {'─'*50}")
    for instr in RATES:
        seg = r_trades[r_trades["instrument"] == instr]
        if seg.empty:
            print(f"  {instr:<10s}  — no trades —")
            continue
        r   = seg["r_multiple"].dropna().values
        pnl = seg["pnl"].sum()
        exp = float(np.mean(r)) if len(r) else float("nan")
        wr  = 100.0 * float(np.mean(r > 0)) if len(r) else float("nan")
        print(f"  {instr:<10s} {len(seg):>7d} {exp:>+7.3f}R  {wr:>5.1f}%  {pnl:>+11.0f}$")

    os.makedirs("output/rates_fix", exist_ok=True)
    eq.to_csv("output/rates_fix/equity_curve.csv")
    trades.to_csv("output/rates_fix/trades.csv", index=False)

    print(f"\n{'─'*W}")
    print(f"  Outputs: output/rates_fix/")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("=" * W)

if __name__ == "__main__":
    main()
