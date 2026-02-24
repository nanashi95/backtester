"""
sector_cap_backtest.py
----------------------
Sector risk cap structural test on the speed-differentiated Donchian ensemble.

System:
  Metals/Energy   : B = Donchian(50)   |  C = Donchian(100)
  Equity/Agri     : B = Donchian(75)   |  C = Donchian(150)
  Rates           : B = Donchian(150)  |  C = Donchian(300)
  No FX.  No volatility scaling.

Sector cap: 30% of total portfolio cap per asset class.
  Total portfolio cap = 6R → sector cap = 1.8R per class.
  Since each trade risks 1R, this allows at most 1 concurrent trade per sector.
  New entries are gated; no resizing or rebalancing.

Baseline (speed-diff, no sector caps):
  CAGR +9.69%  MaxDD -30.98%  MAR 0.313  Sharpe 0.50  Longest UW 2010d

Usage:
    python3 sector_cap_backtest.py
"""

from __future__ import annotations

import os
import sys
import time

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

# Bucket names as returned by get_instrument_bucket()
BUCKET_LABELS = {
    "rates":       "Rates",
    "equity":      "Equity",
    "metals":      "Metals",
    "energy":      "Energy",
    "agriculture": "Agriculture",
}
BUCKET_ORDER = ["rates", "equity", "metals", "energy", "agriculture"]

ASSET_CLASS_MAP: dict[str, str] = {}
for _i in RATES:       ASSET_CLASS_MAP[_i] = "rates"
for _i in EQUITY:      ASSET_CLASS_MAP[_i] = "equity"
for _i in METALS:      ASSET_CLASS_MAP[_i] = "metals"
for _i in ENERGY:      ASSET_CLASS_MAP[_i] = "energy"
for _i in AGRICULTURE: ASSET_CLASS_MAP[_i] = "agriculture"

# ── Risk ──────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0
SECTOR_CAP_PCT = 0.30          # 30% of 6R = 1.8R per sector
START          = "2008-01-01"
END            = "2025-12-31"

W = 72

# ── Baseline (speed-diff, no sector caps, no vol-scaling) ─────────────────────

BASELINE = {
    "cagr":                    0.0969,
    "max_drawdown":           -0.3098,
    "mar_ratio":               0.313,
    "sharpe":                  0.50,
    "longest_underwater_days": 2010,
}

# ── Strategies ────────────────────────────────────────────────────────────────

def make_strategies():
    return {
        "B_ME": make_sleeve(50,  "B"),
        "C_ME": make_sleeve(100, "C"),
        "B_EA": make_sleeve(75,  "B"),
        "C_EA": make_sleeve(150, "C"),
        "B_R":  make_sleeve(150, "B"),
        "C_R":  make_sleeve(300, "C"),
    }

SLEEVE_INSTRUMENTS = {
    "B_ME": METALS_ENERGY,
    "C_ME": METALS_ENERGY,
    "B_EA": EQUITY_AGRI,
    "C_EA": EQUITY_AGRI,
    "B_R":  RATES,
    "C_R":  RATES,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v == v else "nan"


def ann_vol_pct(equity_curve: pd.DataFrame) -> float:
    eq = equity_curve["total_value"]
    eq_d = eq.resample("D").last().ffill()
    dr   = eq_d.pct_change().dropna()
    return float(dr.std() * np.sqrt(252) * 100) if len(dr) > 1 else float("nan")


def compute_daily_sector_exposure(trades_df: pd.DataFrame,
                                  start: str, end: str) -> pd.DataFrame:
    """
    For each sector, compute daily R exposure (# open trades) via cumsum events.
    Returns DataFrame indexed by business-day date range.
    """
    dates = pd.bdate_range(start, end)
    result = {}
    for bkt in BUCKET_ORDER:
        seg = trades_df[trades_df["bucket"] == bkt].copy()
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


# ── Section printers ──────────────────────────────────────────────────────────

def print_portfolio_metrics(m: dict, eq: pd.DataFrame, label: str):
    n_yr = m.get("n_years", 1)
    tr   = m.get("total_trades", 0)
    vol  = ann_vol_pct(eq)
    print(f"\n{'═'*W}")
    print(f"  PORTFOLIO METRICS — {label}")
    print(f"{'═'*W}")
    print(f"  CAGR              : {m.get('cagr', 0)*100:+.2f}%")
    print(f"  Max Drawdown      : {m.get('max_drawdown', 0)*100:.2f}%")
    print(f"  MAR               : {fmt_f(m.get('mar_ratio', float('nan')), 3)}")
    print(f"  Sharpe            : {fmt_f(m.get('sharpe', float('nan')), 2)}")
    print(f"  Profit Factor     : {fmt_f(m.get('profit_factor', float('nan')), 2)}")
    print(f"  Longest UW        : {m.get('longest_underwater_days', 0)} days")
    print(f"  Ann. Volatility   : {fmt_f(vol, 2)}%")
    print(f"  Total Trades      : {tr}  ({tr/n_yr if n_yr else 0:.0f}/yr)")
    print(f"  Win Rate          : {m.get('win_rate', 0)*100:.1f}%")
    print(f"  Expectancy        : {fmt_f(m.get('expectancy_r', 0), 3)}R")


def print_comparison(m_new: dict):
    print(f"\n{'─'*W}")
    print(f"  DELTA vs SPEED-DIFF BASELINE (no sector caps)")
    print(f"{'─'*W}")
    print(f"  {'Metric':<22s} {'New':>10s}  {'Baseline':>10s}  {'Delta':>10s}")
    print(f"  {'─'*58}")

    def row(label, key, scale, unit):
        nv = m_new.get(key, float("nan"))
        bv = BASELINE.get(key, float("nan"))
        if nv == nv and bv == bv:
            d   = (nv - bv) * scale
            nv_s = f"{nv * scale:.2f}{unit}"
            bv_s = f"{bv * scale:.2f}{unit}"
            d_s  = f"{'+' if d >= 0 else ''}{d:.2f}{unit}"
        else:
            nv_s = bv_s = d_s = "nan"
        print(f"  {label:<22s} {nv_s:>10s}  {bv_s:>10s}  {d_s:>10s}")

    row("CAGR",       "cagr",                    100, "%")
    row("MaxDD",      "max_drawdown",             100, "%")
    row("MAR",        "mar_ratio",                  1, "")
    row("Sharpe",     "sharpe",                     1, "")
    row("Longest UW", "longest_underwater_days",    1, "d")


def print_asset_class(trades_df: pd.DataFrame):
    print(f"\n{'─'*W}")
    print(f"  ASSET CLASS BREAKDOWN — 2008–2025")
    print(f"{'─'*W}")
    total_pnl = trades_df["pnl"].sum()
    print(f"  {'Class':<14s} {'Trades':>6s} {'Expect':>8s} {'AvgHold':>8s} "
          f"{'Win%':>6s} {'TotalPnL':>12s}")
    print(f"  {'─'*62}")
    for bkt in BUCKET_ORDER:
        seg = trades_df[trades_df["bucket"] == bkt]
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


def print_concentration_diagnostics(trades_df: pd.DataFrame,
                                    start: str, end: str):
    print(f"\n{'─'*W}")
    print(f"  CONCENTRATION DIAGNOSTICS")
    print(f"{'─'*W}")
    exposure = compute_daily_sector_exposure(trades_df, start, end)

    print(f"\n  {'Sector':<14s} {'Avg R Open':>11s} {'Max R Spike':>12s} "
          f"{'%PnL Contrib':>13s} {'Trades':>7s}")
    print(f"  {'─'*60}")

    total_pnl = trades_df["pnl"].sum()
    for bkt in BUCKET_ORDER:
        lbl = BUCKET_LABELS[bkt]
        seg = trades_df[trades_df["bucket"] == bkt]
        avg_r = float(exposure[bkt].mean())
        max_r = float(exposure[bkt].max())
        pnl   = seg["pnl"].sum() if not seg.empty else 0.0
        pct   = pnl / total_pnl * 100 if total_pnl else 0.0
        n     = len(seg)
        print(f"  {lbl:<14s}"
              f" {avg_r:>10.2f}R"
              f"  {max_r:>10.1f}R"
              f"  {pct:>+11.1f}%"
              f" {n:>7d}")

    # Portfolio-level stats
    total_r = exposure.sum(axis=1)
    print(f"\n  Portfolio avg R open : {total_r.mean():.2f}R")
    print(f"  Portfolio max R open : {total_r.max():.1f}R")
    print(f"  Days ≥ 4R open       : {(total_r >= 4).mean()*100:.1f}%")


def print_rejection_stats(rejections: dict, trades_total: int):
    print(f"\n{'─'*W}")
    print(f"  SECTOR CAP REJECTIONS  "
          f"(cap = {SECTOR_CAP_PCT*100:.0f}% × {GLOBAL_CAP:.0f}R = "
          f"{GLOBAL_CAP*SECTOR_CAP_PCT:.1f}R per sector)")
    print(f"{'─'*W}")

    total_rej = sum(rejections.values())
    total_sig = trades_total + total_rej   # attempted = accepted + rejected

    print(f"\n  {'Sector':<14s} {'Rejected':>9s} {'% of Attempted':>15s}")
    print(f"  {'─'*42}")
    for bkt in BUCKET_ORDER:
        lbl = BUCKET_LABELS[bkt]
        n_rej = rejections.get(bkt, 0)
        # Attempted = trades in this bucket + rejections in this bucket
        n_acc = len([1 for _ in range(0)])  # placeholder; computed below
        print(f"  {lbl:<14s}  {n_rej:>8d}")

    print(f"\n  Total rejected : {total_rej}")
    print(f"  Total accepted : {trades_total}")
    print(f"  Rejection rate : {total_rej / (total_rej + trades_total) * 100:.1f}%  "
          f"(of all attempted entries)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    sector_cap_r = GLOBAL_CAP * SECTOR_CAP_PCT

    print("=" * W)
    print("  SECTOR CAP BACKTEST — Speed-Diff Donchian + 30% Sector Risk Cap")
    print(f"  Universe : {len(ALL_NO_FX)} instruments  "
          f"(Rates / Equity / Metals / Energy / Agriculture)")
    print(f"  Sleeves  : B_ME=D50  C_ME=D100 | B_EA=D75  C_EA=D150 | B_R=D150  C_R=D300")
    print(f"  Sector   : max {SECTOR_CAP_PCT*100:.0f}% × {GLOBAL_CAP:.0f}R = "
          f"{sector_cap_r:.1f}R per class  (≈ {int(sector_cap_r)} trade at a time)")
    print(f"  Risk     : {RISK_PER_TRADE*100:.1f}%/tr  "
          f"1.5×ATR init  2×ATR trail  {GLOBAL_CAP:.0f}R global cap")
    print("=" * W)

    strategies = make_strategies()

    eng = EnsemblePortfolioEngine(
        strategies         = strategies,
        start              = START,
        end                = END,
        initial_equity     = INITIAL_EQUITY,
        instruments        = ALL_NO_FX,
        risk_per_trade     = RISK_PER_TRADE,
        global_cap         = GLOBAL_CAP,
        sleeve_instruments = SLEEVE_INSTRUMENTS,
        sector_cap_pct     = SECTOR_CAP_PCT,
        use_vol_scaling    = False,
    )
    eq         = eng.run()
    trades     = eng.get_trades_df()
    rejections = eng.get_sector_cap_rejections()
    metrics    = compute_all_metrics(eq, trades, INITIAL_EQUITY)

    print_portfolio_metrics(metrics, eq, "2008–2025  Speed-Diff + Sector Cap 30%")
    print_comparison(metrics)
    print_asset_class(trades)
    print_concentration_diagnostics(trades, START, END)
    print_rejection_stats(rejections, metrics.get("total_trades", 0))

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs("output/sector_cap", exist_ok=True)
    eq.to_csv("output/sector_cap/equity_curve.csv")
    trades.to_csv("output/sector_cap/trades.csv", index=False)

    print(f"\n{'─'*W}")
    print(f"  Outputs: output/sector_cap/  (equity_curve.csv, trades.csv)")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
