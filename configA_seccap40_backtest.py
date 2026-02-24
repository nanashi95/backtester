"""
configA_seccap40_backtest.py
----------------------------
Config A + 40% sector cap.

  Metals/Energy : B=D50   C=D100
  Equity/Agri   : B=D75   C=D150
  Rates (ZN+ZF) : B=D50   C=D100   (ZB excluded)
  Sector cap    : 40% × 6R = 2.4R per class  (max 2 concurrent trades/sector)
  No FX  |  No vol-scaling

Baselines:
  Speed-diff original : CAGR +9.69%  MaxDD -30.98%  MAR 0.313  Sharpe 0.50
  Config A uncapped   : CAGR +10.57% MaxDD -31.10%  MAR 0.340  Sharpe 0.54
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

RATES         = ["ZN", "ZF"]               # ZB excluded
EQUITY        = ["ES", "NQ", "RTY", "YM"]
METALS        = ["GC", "SI", "HG", "PA", "PL"]
ENERGY        = ["CL", "NG"]
AGRICULTURE   = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

METALS_ENERGY = METALS + ENERGY
EQUITY_AGRI   = EQUITY + AGRICULTURE
ALL_INSTR     = RATES + EQUITY + METALS + ENERGY + AGRICULTURE

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
SECTOR_CAP_PCT = 0.40          # 40% × 6R = 2.4R → max 2 trades per sector
START, END     = "2008-01-01", "2025-12-31"
W = 72

BASELINES = {
    "Speed-diff orig": {
        "cagr": 0.0969, "max_drawdown": -0.3098,
        "mar_ratio": 0.313, "sharpe": 0.50,
        "longest_underwater_days": 2010,
    },
    "Config A uncapped": {
        "cagr": 0.1057, "max_drawdown": -0.3110,
        "mar_ratio": 0.340, "sharpe": 0.54,
        "longest_underwater_days": 1911,
    },
}

PERIODS = [
    ("2008-01-01", "2016-12-31", "2008–2016"),
    ("2017-01-01", "2022-12-31", "2017–2022"),
    ("2023-01-01", "2025-12-31", "2023–2025"),
    ("2008-01-01", "2025-12-31", "2008–2025"),
]

# ── Setup ─────────────────────────────────────────────────────────────────────

def make_strategies():
    return {
        "B_ME": make_sleeve(50,  "B"),
        "C_ME": make_sleeve(100, "C"),
        "B_EA": make_sleeve(75,  "B"),
        "C_EA": make_sleeve(150, "C"),
        "B_R":  make_sleeve(50,  "B"),
        "C_R":  make_sleeve(100, "C"),
    }

SLEEVE_INSTRUMENTS = {
    "B_ME": METALS_ENERGY, "C_ME": METALS_ENERGY,
    "B_EA": EQUITY_AGRI,   "C_EA": EQUITY_AGRI,
    "B_R":  RATES,         "C_R":  RATES,
}

def run(start, end):
    eng = EnsemblePortfolioEngine(
        strategies         = make_strategies(),
        start              = start,
        end                = end,
        initial_equity     = INITIAL_EQUITY,
        instruments        = ALL_INSTR,
        risk_per_trade     = RISK_PER_TRADE,
        global_cap         = GLOBAL_CAP,
        sleeve_instruments = SLEEVE_INSTRUMENTS,
        sector_cap_pct     = SECTOR_CAP_PCT,
        use_vol_scaling    = False,
    )
    eq     = eng.run()
    trades = eng.get_trades_df()
    rejections = eng.get_sector_cap_rejections()
    return eq, trades, rejections

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v == v else "nan"

def ann_vol(eq):
    s = eq["total_value"].resample("D").last().ffill()
    dr = s.pct_change().dropna()
    return float(dr.std() * np.sqrt(252) * 100) if len(dr) > 1 else float("nan")

def compute_daily_sector_exposure(trades_df, start, end):
    dates = pd.bdate_range(start, end)
    result = {}
    for bkt in BUCKET_ORDER:
        seg = trades_df[trades_df["bucket"] == bkt]
        events = pd.Series(0.0, index=dates)
        for _, t in seg.iterrows():
            entry = pd.Timestamp(t["entry_date"])
            exit_ = pd.Timestamp(t["exit_date"])
            if entry in events.index:
                events.loc[entry] += 1.0
            nxt = exit_ + pd.offsets.BDay(1)
            if nxt in events.index:
                events.loc[nxt] -= 1.0
        result[bkt] = events.cumsum().clip(lower=0)
    return pd.DataFrame(result)

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
    print(f"  Ann. Volatility   : {fmt_f(ann_vol(eq), 2)}%")
    print(f"  Total Trades      : {tr}  ({tr/n_yr:.0f}/yr)")
    print(f"  Win Rate          : {m.get('win_rate', 0)*100:.1f}%")
    print(f"  Expectancy        : {fmt_f(m.get('expectancy_r', 0), 3)}R")

def print_comparison(m_new):
    print(f"\n{'─'*W}")
    print(f"  DELTA vs BASELINES")
    print(f"{'─'*W}")
    col = 16
    print(f"  {'Metric':<22s}", end="")
    print(f"  {'This run':>{col}s}", end="")
    for lbl in BASELINES:
        print(f"  {lbl[:col]:>{col}s}", end="")
    print()
    print(f"  {'─'*(22 + (col+2) * (len(BASELINES)+1))}")

    rows = [
        ("CAGR",       "cagr",                    100, "%"),
        ("MaxDD",      "max_drawdown",             100, "%"),
        ("MAR",        "mar_ratio",                  1, ""),
        ("Sharpe",     "sharpe",                     1, ""),
        ("Longest UW", "longest_underwater_days",    1, "d"),
    ]
    for label, key, scale, unit in rows:
        nv = m_new.get(key, float("nan"))
        nv_s = f"{nv*scale:.2f}{unit}" if nv==nv else "nan"
        print(f"  {label:<22s}  {nv_s:>{col}s}", end="")
        for bname, bvals in BASELINES.items():
            bv = bvals.get(key, float("nan"))
            if nv==nv and bv==bv:
                d = (nv - bv) * scale
                d_s = f"{'+' if d>=0 else ''}{d:.2f}{unit}"
                print(f"  {d_s:>{col}s}", end="")
            else:
                print(f"  {'nan':>{col}s}", end="")
        print()

def print_asset_class(trades, label="2008–2025"):
    print(f"\n{'─'*W}")
    print(f"  ASSET CLASS — {label}")
    print(f"{'─'*W}")
    total_pnl = trades["pnl"].sum()
    print(f"  {'Class':<14s} {'Trades':>6s} {'Expect':>8s} "
          f"{'AvgHold':>8s} {'Win%':>6s} {'TotalPnL':>12s}")
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
        print(f"  {lbl:<14s}"
              f" {len(seg):>6d}"
              f" {exp:>+7.3f}R"
              f"  {hold:>6.1f}d"
              f" {wr:>5.1f}%"
              f" {pnl:>+11.0f}$  ({pct:>+5.1f}%)")

def print_regime_table(period_results):
    print(f"\n{'─'*W}")
    print(f"  REGIME TABLE")
    print(f"{'─'*W}")
    print(f"  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  "
          f"{'MAR':>6s}  {'Sharpe':>6s}  {'Trades/yr':>10s}")
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

def print_concentration(trades, eq):
    print(f"\n{'─'*W}")
    print(f"  CONCENTRATION & EXPOSURE — 2008–2025")
    print(f"{'─'*W}")
    exposure = compute_daily_sector_exposure(trades, START, END)
    total_pnl = trades["pnl"].sum()

    print(f"\n  {'Sector':<14s} {'Avg R':>7s} {'Max R':>7s} "
          f"{'%PnL':>8s} {'Trades':>7s}")
    print(f"  {'─'*48}")
    for bkt in BUCKET_ORDER:
        lbl = BUCKET_LABELS[bkt]
        seg = trades[trades["bucket"] == bkt]
        avg_r = float(exposure[bkt].mean())
        max_r = float(exposure[bkt].max())
        pnl   = seg["pnl"].sum() if not seg.empty else 0.0
        pct   = pnl / total_pnl * 100 if total_pnl else 0.0
        n     = len(seg)
        print(f"  {lbl:<14s}  {avg_r:>5.2f}R  {max_r:>5.1f}R  "
              f"{pct:>+6.1f}%  {n:>6d}")

    total_r = exposure.sum(axis=1)
    ec = eq["total_r_open"]
    print(f"\n  Portfolio avg R open : {total_r.mean():.2f}R  "
          f"(equity curve: {ec.mean():.2f}R)")
    print(f"  Portfolio max R open : {total_r.max():.1f}R")
    print(f"  Days ≥ 3R open       : {(total_r >= 3).mean()*100:.1f}%")
    print(f"  Days ≥ 4R open       : {(total_r >= 4).mean()*100:.1f}%")

def print_rejections(rejections, n_accepted):
    print(f"\n{'─'*W}")
    print(f"  SECTOR CAP REJECTIONS  "
          f"(40% × {GLOBAL_CAP:.0f}R = {GLOBAL_CAP*SECTOR_CAP_PCT:.1f}R/sector)")
    print(f"{'─'*W}")
    total_rej = sum(rejections.values())
    print(f"\n  {'Sector':<14s}  {'Rejected':>8s}")
    print(f"  {'─'*26}")
    for bkt in BUCKET_ORDER:
        lbl = BUCKET_LABELS[bkt]
        print(f"  {lbl:<14s}  {rejections.get(bkt, 0):>8d}")
    print(f"\n  Total rejected : {total_rej}")
    print(f"  Total accepted : {n_accepted}")
    rej_rate = total_rej / (total_rej + n_accepted) * 100 if (total_rej + n_accepted) > 0 else 0
    print(f"  Rejection rate : {rej_rate:.1f}%")
    print(f"\n  (30% cap had 87.3% rejection — both slow sleeves eliminated)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    sector_cap_r = GLOBAL_CAP * SECTOR_CAP_PCT
    print("=" * W)
    print("  CONFIG A + 40% SECTOR CAP")
    print(f"  Rates  : ZN + ZF,  B=D50  C=D100  (ZB excluded)")
    print(f"  Metals : B=D50  C=D100  |  Equity/Agri: B=D75  C=D150")
    print(f"  Cap    : {SECTOR_CAP_PCT*100:.0f}% × {GLOBAL_CAP:.0f}R = "
          f"{sector_cap_r:.1f}R/sector  (max 2 trades per class)")
    print(f"  Risk   : {RISK_PER_TRADE*100:.1f}%/tr  "
          f"1.5×ATR init  2×ATR trail  {GLOBAL_CAP:.0f}R global cap")
    print("=" * W)

    eq, trades, rejections = run(START, END)
    metrics = compute_all_metrics(eq, trades, INITIAL_EQUITY)

    print_metrics(metrics, eq, "2008–2025  Config A + 40% Sector Cap")
    print_comparison(metrics)
    print_asset_class(trades)

    # Per-period
    period_results = []
    for start, end, label in PERIODS:
        print(f"\n  Running {label}...")
        eq_p, tr_p, _ = run(start, end)
        m_p = compute_all_metrics(eq_p, tr_p, INITIAL_EQUITY)
        period_results.append((label, m_p))
    print_regime_table(period_results)

    print_concentration(trades, eq)
    print_rejections(rejections, metrics.get("total_trades", 0))

    os.makedirs("output/configA_seccap40", exist_ok=True)
    eq.to_csv("output/configA_seccap40/equity_curve.csv")
    trades.to_csv("output/configA_seccap40/trades.csv", index=False)

    print(f"\n{'─'*W}")
    print(f"  Outputs: output/configA_seccap40/")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("=" * W)

if __name__ == "__main__":
    main()
