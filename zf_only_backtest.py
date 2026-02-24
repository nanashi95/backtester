"""
zf_only_backtest.py
───────────────────
Config A variant: ZF only in rates (drop ZN + ZB).
Per-instrument: ZF +$24k, ZN -$655 in rates_fix run → test ZF alone.

Rates: B=D50, C=D100. All other sleeves unchanged.
"""

from __future__ import annotations
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import data.loader as loader
import data.yf_data_loader as yf_loader
loader.configure(yf_loader.load_all_data, yf_loader.get_instrument_bucket)

from strategies.donchian_ensemble import make_sleeve
from engine.ensemble_engine import EnsemblePortfolioEngine
from metrics.metrics_engine import compute_all_metrics

# ── Universe ──────────────────────────────────────────────────────────────────

RATES       = ["ZF"]                          # ZF only
EQUITY      = ["ES", "NQ", "RTY", "YM"]
METALS      = ["GC", "SI", "HG", "PA", "PL"]
ENERGY      = ["CL", "NG"]
AGRICULTURE = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

ALL_INSTR = RATES + EQUITY + METALS + ENERGY + AGRICULTURE

SLEEVE_INSTRUMENTS = {
    "B_ME": METALS + ENERGY,
    "C_ME": METALS + ENERGY,
    "B_EA": EQUITY + AGRICULTURE,
    "C_EA": EQUITY + AGRICULTURE,
    "B_R":  RATES,
    "C_R":  RATES,
}

# ── Risk ──────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0
START, END     = "2008-01-01", "2025-12-31"
W = 72

BASELINES = {
    "Speed-diff orig":   {"cagr": 0.0969, "max_drawdown": -0.3098, "mar_ratio": 0.313, "sharpe": 0.50, "longest_underwater_days": 2010},
    "Config A (ZN+ZF)":  {"cagr": 0.1057, "max_drawdown": -0.3110, "mar_ratio": 0.340, "sharpe": 0.54, "longest_underwater_days": 1911},
}

PERIODS = [
    ("2008-01-01", "2016-12-31", "2008–2016"),
    ("2017-01-01", "2022-12-31", "2017–2022"),
    ("2023-01-01", "2025-12-31", "2023–2025"),
    ("2008-01-01", "2025-12-31", "2008–2025"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_strategies():
    return {
        "B_ME": make_sleeve(50,  "B"),
        "C_ME": make_sleeve(100, "C"),
        "B_EA": make_sleeve(75,  "B"),
        "C_EA": make_sleeve(150, "C"),
        "B_R":  make_sleeve(50,  "B"),
        "C_R":  make_sleeve(100, "C"),
    }


def run_period(start, end, label):
    print(f"\n  Running {label}...")
    eng = EnsemblePortfolioEngine(
        strategies         = make_strategies(),
        start              = start,
        end                = end,
        initial_equity     = INITIAL_EQUITY,
        instruments        = ALL_INSTR,
        risk_per_trade     = RISK_PER_TRADE,
        global_cap         = GLOBAL_CAP,
        sleeve_instruments = SLEEVE_INSTRUMENTS,
        sector_cap_pct     = 0.0,
        use_vol_scaling    = False,
    )
    eq     = eng.run()
    trades = eng.get_trades_df()
    m      = compute_all_metrics(eq, trades, INITIAL_EQUITY)
    return m, trades, eq


# ── Header ────────────────────────────────────────────────────────────────────

t0 = time.time()
print("=" * W)
print("  ZF-ONLY RATES")
print("  Rates  : ZF only,  B=D50  C=D100  (ZN + ZB excluded)")
print("  Metals : B=D50  C=D100  |  Equity/Agri: B=D75  C=D150")
print("  Risk   : 0.7%/tr  1.5×ATR init  2×ATR trail  6R global cap")
print("=" * W)

# ── Full-period detailed run ──────────────────────────────────────────────────

m_full, trades_full, eq_full = run_period(START, END, "2008–2025")

n_trades = len(trades_full)

print(f"\n{'═'*W}")
print(f"  PORTFOLIO METRICS — 2008–2025  ZF-only rates")
print(f"{'═'*W}")
print(f"  CAGR              : {m_full['cagr']:+.2%}")
print(f"  Max Drawdown      : {m_full['max_drawdown']:.2%}")
print(f"  MAR               : {m_full['mar_ratio']:.3f}")
print(f"  Sharpe            : {m_full['sharpe']:.2f}")
print(f"  Profit Factor     : {m_full.get('profit_factor', float('nan')):.2f}")
print(f"  Longest UW        : {m_full['longest_underwater_days']} days")
print(f"  Total Trades      : {n_trades}  ({n_trades/17:.0f}/yr)")
print(f"  Win Rate          : {m_full.get('win_rate', float('nan')):.1%}")
print(f"  Expectancy        : {m_full.get('expectancy', float('nan')):.3f}R")

# ── Delta vs baselines ────────────────────────────────────────────────────────

print(f"\n{'─'*W}")
print(f"  DELTA vs BASELINES")
print(f"{'─'*W}")
metrics_rows = [
    ("CAGR",      "cagr",          m_full["cagr"],          ".2%"),
    ("MaxDD",     "max_drawdown",  m_full["max_drawdown"],  ".2%"),
    ("MAR",       "mar_ratio",     m_full["mar_ratio"],     ".3f"),
    ("Sharpe",    "sharpe",        m_full["sharpe"],        ".2f"),
    ("LongestUW", "longest_underwater_days", m_full["longest_underwater_days"], ".0f"),
]
bnames = list(BASELINES.keys())
col_w  = 18
hdr = f"  {'Metric':<12}  {'This run':>10}  " + "  ".join(f"{b[:col_w]:>{col_w}}" for b in bnames)
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for lbl, key, val, fmt in metrics_rows:
    deltas = []
    for bname in bnames:
        bv = BASELINES[bname][key]
        d  = val - bv
        sign = "+" if d >= 0 else ""
        deltas.append(f"{sign}{d:{fmt}}")
    print(f"  {lbl:<12}  {val:>{10}{fmt}}  " + "  ".join(f"{d:>{col_w}}" for d in deltas))

# ── Asset class breakdown ─────────────────────────────────────────────────────

print(f"\n{'─'*W}")
print(f"  ASSET CLASS — 2008–2025")
print(f"{'─'*W}")
print(f"  {'Class':<14} {'Trades':>7}  {'Expect':>8}  {'AvgHold':>8}  {'Win%':>5}  {'TotalPnL':>14}")
print(f"  {'─'*66}")
total_pnl = trades_full["pnl"].sum()
for bucket, grp in trades_full.groupby("bucket"):
    n        = len(grp)
    exp      = grp["r_multiple"].mean()
    avg_hold = grp["hold_days"].mean()
    wr       = (grp["r_multiple"] > 0).mean()
    pnl      = grp["pnl"].sum()
    pct      = pnl / total_pnl * 100 if total_pnl != 0 else 0
    sign     = "+" if pnl >= 0 else ""
    print(f"  {bucket.capitalize():<14} {n:>7}  {exp:>+.3f}R  {avg_hold:>7.1f}d  "
          f"{wr:>4.1%}  {sign}${pnl:>9,.0f}  ({pct:>+.1f}%)")

# ── Regime table ──────────────────────────────────────────────────────────────

print(f"\n{'─'*W}")
print(f"  REGIME TABLE")
print(f"{'─'*W}")
print(f"  {'Period':<20} {'CAGR':>8}  {'MaxDD':>8}  {'MAR':>6}  {'Sharpe':>7}  {'Trades/yr':>10}")
print(f"  {'─'*66}")

for pstart, pend, plabel in PERIODS:
    m_p, tr_p, _ = run_period(pstart, pend, plabel)
    yrs = (pd.Timestamp(pend) - pd.Timestamp(pstart)).days / 365.25
    tpy = len(tr_p) / yrs
    print(f"  {plabel:<20} {m_p['cagr']:>+.2%}  {m_p['max_drawdown']:>8.2%}  "
          f"{m_p['mar_ratio']:>6.3f}  {m_p['sharpe']:>7.2f}  {tpy:>8.0f}/yr")

elapsed = time.time() - t0
print(f"\n{'─'*W}")
print(f"  Elapsed: {elapsed:.1f}s")
print("=" * W)
