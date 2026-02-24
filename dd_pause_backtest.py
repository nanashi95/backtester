"""
dd_pause_backtest.py
────────────────────
Config A (ZN+ZF, D50/D100, ZB excluded) + 15% equity drawdown pause.

When MTM equity falls >15% below its high-water mark, no new trades are
opened until the portfolio recovers. Existing trades continue to run normally.
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

RATES       = ["ZN", "ZF"]               # ZB excluded
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

# ── Settings ──────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0
DD_PAUSE_PCT   = 0.15     # 15% from high-water mark

BASELINES = {
    "Speed-diff orig":   {"cagr": 0.0969, "max_drawdown": -0.3098, "mar_ratio": 0.313, "sharpe": 0.50, "longest_underwater_days": 2010},
    "Config A uncapped": {"cagr": 0.1057, "max_drawdown": -0.3110, "mar_ratio": 0.340, "sharpe": 0.54, "longest_underwater_days": 1911},
}

OUT_DIR = "output/dd_pause"
os.makedirs(OUT_DIR, exist_ok=True)

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
        dd_pause_pct       = DD_PAUSE_PCT,
    )
    eq        = eng.run()
    trades_df = eng.get_trades_df()
    metrics   = compute_all_metrics(eq, trades_df, INITIAL_EQUITY)
    pause_days = eng._dd_pause_days
    total_days = len(eq)
    return metrics, trades_df, eq, pause_days, total_days


# ── Header ────────────────────────────────────────────────────────────────────

t0 = time.time()
print("=" * 72)
print("  CONFIG A + 15% DRAWDOWN PAUSE")
print("  Rates  : ZN + ZF,  B=D50  C=D100  (ZB excluded)")
print("  Metals : B=D50  C=D100  |  Equity/Agri: B=D75  C=D150")
print(f"  DD gate: pause new entries when MTM falls >{DD_PAUSE_PCT:.0%} from HWM")
print("  Risk   : 0.7%/tr  1.5×ATR init  2×ATR trail  6R global cap")
print("=" * 72)

# ── Full-period run for detailed output ───────────────────────────────────────

m_full, trades_full, eq_full, pause_days_full, total_days_full = run_period(
    "2008-01-01", "2025-12-31", "2008–2025"
)

# ── Summary metrics ───────────────────────────────────────────────────────────

print(f"\n{'═'*72}")
print(f"  PORTFOLIO METRICS — 2008–2025  Config A + {DD_PAUSE_PCT:.0%} DD Pause")
print(f"{'═'*72}")
print(f"  CAGR              : {m_full['cagr']:+.2%}")
print(f"  Max Drawdown      : {m_full['max_drawdown']:.2%}")
print(f"  MAR               : {m_full['mar_ratio']:.3f}")
print(f"  Sharpe            : {m_full['sharpe']:.2f}")
pf = m_full.get('profit_factor', float('nan'))
print(f"  Profit Factor     : {pf:.2f}")
print(f"  Longest UW        : {m_full['longest_underwater_days']} days")
av = m_full.get('annual_volatility', float('nan'))
print(f"  Ann. Volatility   : {av:.2%}")
n_trades = len(trades_full)
print(f"  Total Trades      : {n_trades}  ({n_trades/17:.0f}/yr)")
wr = m_full.get('win_rate', float('nan'))
print(f"  Win Rate          : {wr:.1%}")
exp = m_full.get('expectancy', float('nan'))
print(f"  Expectancy        : {exp:.3f}R")
print(f"\n  DD-pause stats:")
print(f"    Days paused      : {pause_days_full}  ({pause_days_full/total_days_full:.1%} of trading days)")

# ── Delta vs baselines ────────────────────────────────────────────────────────

print(f"\n{'─'*72}")
print(f"  DELTA vs BASELINES")
print(f"{'─'*72}")
col_w   = 18
metrics_rows = [
    ("CAGR",      "cagr",          m_full["cagr"],          ".2%"),
    ("MaxDD",     "max_drawdown",  m_full["max_drawdown"],  ".2%"),
    ("MAR",       "mar_ratio",     m_full["mar_ratio"],     ".3f"),
    ("Sharpe",    "sharpe",        m_full["sharpe"],        ".2f"),
    ("LongestUW", "longest_underwater_days", m_full["longest_underwater_days"], ".0f"),
]
bnames = list(BASELINES.keys())
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
    row_val = f"{val:{fmt}}"
    print(f"  {lbl:<12}  {row_val:>10}  " + "  ".join(f"{d:>{col_w}}" for d in deltas))

# ── Asset class breakdown ─────────────────────────────────────────────────────

print(f"\n{'─'*72}")
print(f"  ASSET CLASS — 2008–2025")
print(f"{'─'*72}")
print(f"  {'Class':<14} {'Trades':>7}  {'Expect':>8}  {'AvgHold':>8}  {'Win%':>5}  {'TotalPnL':>14}")
print(f"  {'─'*68}")
total_pnl = trades_full["pnl"].sum()
for bucket, grp in trades_full.groupby("bucket"):
    n        = len(grp)
    exp_b    = grp["r_multiple"].mean()
    avg_hold = grp["hold_days"].mean()
    wr_b     = (grp["r_multiple"] > 0).mean()
    pnl      = grp["pnl"].sum()
    pct      = pnl / total_pnl * 100 if total_pnl != 0 else 0
    sign     = "+" if pnl >= 0 else ""
    print(f"  {bucket.capitalize():<14} {n:>7}  {exp_b:>+.3f}R  {avg_hold:>7.1f}d  "
          f"{wr_b:>4.1%}  {sign}${pnl:>9,.0f}  ({pct:>+.1f}%)")

# ── Regime table ──────────────────────────────────────────────────────────────

print(f"\n{'─'*72}")
print(f"  REGIME TABLE")
print(f"{'─'*72}")
print(f"  {'Period':<20} {'CAGR':>8}  {'MaxDD':>8}  {'MAR':>6}  {'Sharpe':>7}  {'Trades/yr':>10}  {'Pause%':>7}")
print(f"  {'─'*72}")

for pstart, pend, plabel in PERIODS:
    m_p, tr_p, eq_p, pd_p, td_p = run_period(pstart, pend, plabel)
    yrs       = (pd.Timestamp(pend) - pd.Timestamp(pstart)).days / 365.25
    tpy       = len(tr_p) / yrs
    pause_pct = pd_p / td_p if td_p > 0 else 0.0
    print(f"  {plabel:<20} {m_p['cagr']:>+.2%}  {m_p['max_drawdown']:>8.2%}  "
          f"{m_p['mar_ratio']:>6.3f}  {m_p['sharpe']:>7.2f}  {tpy:>8.0f}/yr  {pause_pct:>6.1%}")

elapsed = time.time() - t0
print(f"\n{'─'*72}")
print(f"  Outputs: {OUT_DIR}/")
print(f"  Elapsed: {elapsed:.1f}s")
print("=" * 72)
