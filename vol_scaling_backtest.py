"""
vol_scaling_backtest.py
-----------------------
ATR percentile volatility scaling structural test.

System: Speed-differentiated Donchian ensemble (no FX), 2008–2025.
  Metals/Energy : B = Donchian(50)   |  C = Donchian(100)
  Equity/Agri   : B = Donchian(75)   |  C = Donchian(150)
  Rates         : B = Donchian(150)  |  C = Donchian(300)

Vol-scaling (ATR percentile, 252-day rolling window):
  pct < 0.33  →  1.2× base risk   (low-vol — expand)
  pct 0.33–0.66 → 1.0× base risk  (normal)
  pct > 0.66  →  0.6× base risk   (high-vol — compress)

Position size fixed at entry; stop logic and portfolio caps unchanged.

Baseline (speed-diff, no vol-scaling, from prior session):
  CAGR +9.69%  MaxDD -30.98%  MAR 0.313  Sharpe 0.50  Longest UW 2010d

Usage:
    python3 vol_scaling_backtest.py
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

ASSET_CLASS_MAP = {}
for _i in RATES:       ASSET_CLASS_MAP[_i] = "Rates"
for _i in EQUITY:      ASSET_CLASS_MAP[_i] = "Equity"
for _i in METALS:      ASSET_CLASS_MAP[_i] = "Metals"
for _i in ENERGY:      ASSET_CLASS_MAP[_i] = "Energy"
for _i in AGRICULTURE: ASSET_CLASS_MAP[_i] = "Agriculture"
ASSET_CLASSES = ["Rates", "Equity", "Metals", "Energy", "Agriculture"]

# ── Risk ──────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0
START          = "2008-01-01"
END            = "2025-12-31"

W = 72

# ── Baseline results (speed-diff, no vol-scaling, prior session) ──────────────

BASELINE = {
    "cagr":                    0.0969,
    "max_drawdown":           -0.3098,
    "mar_ratio":               0.313,
    "sharpe":                  0.50,
    "longest_underwater_days": 2010,
}

# ── Speed-differentiated sleeve setup ─────────────────────────────────────────

def make_strategies():
    return {
        "B_ME": make_sleeve(50,  "B"),   # Metals + Energy fast
        "C_ME": make_sleeve(100, "C"),   # Metals + Energy slow
        "B_EA": make_sleeve(75,  "B"),   # Equity + Agriculture fast
        "C_EA": make_sleeve(150, "C"),   # Equity + Agriculture slow
        "B_R":  make_sleeve(150, "B"),   # Rates fast
        "C_R":  make_sleeve(300, "C"),   # Rates slow
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
    eq_daily = eq.resample("D").last().ffill()
    dr = eq_daily.pct_change().dropna()
    return float(dr.std() * np.sqrt(252) * 100) if len(dr) > 1 else float("nan")


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


def print_comparison(m_new: dict, eq_new: pd.DataFrame):
    print(f"\n{'─'*W}")
    print(f"  DELTA vs SPEED-DIFF BASELINE (no vol-scaling)")
    print(f"{'─'*W}")
    print(f"  {'Metric':<22s} {'New':>10s}  {'Baseline':>10s}  {'Delta':>10s}")
    print(f"  {'─'*58}")

    def row(label, key, scale, unit):
        nv = m_new.get(key, float("nan"))
        bv = BASELINE.get(key, float("nan"))
        if nv == nv and bv == bv:
            d = (nv - bv) * scale
            sign = "+" if d >= 0 else ""
            nv_s = f"{nv * scale:.2f}{unit}"
            bv_s = f"{bv * scale:.2f}{unit}"
            d_s  = f"{sign}{d:.2f}{unit}"
        else:
            nv_s = bv_s = d_s = "nan"
        print(f"  {label:<22s} {nv_s:>10s}  {bv_s:>10s}  {d_s:>10s}")

    row("CAGR",        "cagr",                    100, "%")
    row("MaxDD",       "max_drawdown",             100, "%")
    row("MAR",         "mar_ratio",                  1, "")
    row("Sharpe",      "sharpe",                     1, "")
    row("Longest UW",  "longest_underwater_days",    1, "d")


def print_regime_diagnostics(trades_df: pd.DataFrame):
    print(f"\n{'─'*W}")
    print(f"  VOL REGIME DIAGNOSTICS")
    print(f"{'─'*W}")

    df = trades_df[trades_df["r_multiple"].notna()].copy()

    regime_labels = {1.2: "Low-vol  (<33rd pct)", 1.0: "Mid-vol  (33–66th)", 0.6: "High-vol (>66th pct)"}
    regime_order  = [1.2, 1.0, 0.6]

    total = len(df)
    print(f"\n  {'Regime':<22s} {'Trades':>7s} {'%Tot':>6s} "
          f"{'AvgR':>8s} {'Win%':>7s} {'AvgLoss':>9s} {'MaxLoss':>9s}")
    print(f"  {'─'*70}")

    for mult in regime_order:
        seg = df[df["risk_multiplier"] == mult]
        lbl = regime_labels[mult]
        if seg.empty:
            print(f"  {lbl:<22s}   — no trades —")
            continue
        r     = seg["r_multiple"].values
        n     = len(seg)
        pct_n = n / total * 100
        avg_r = float(np.mean(r))
        wr    = 100.0 * float(np.mean(r > 0))
        losses = r[r < 0]
        avg_l = float(np.mean(losses)) if len(losses) else float("nan")
        max_l = float(np.min(losses))  if len(losses) else float("nan")
        print(f"  {lbl:<22s}"
              f" {n:>7d}"
              f" {pct_n:>5.1f}%"
              f" {avg_r:>+7.3f}R"
              f"  {wr:>5.1f}%"
              f" {avg_l:>+8.3f}R"
              f" {max_l:>+8.3f}R")

    # ── Tail-loss compression check ───────────────────────────────────────────
    print(f"\n  TAIL LOSS ANALYSIS")
    print(f"  {'Regime':<22s} {'P5':>8s} {'P1':>8s} {'Min':>8s}")
    print(f"  {'─'*50}")
    all_r = df["r_multiple"].values
    p5_all = float(np.percentile(all_r, 5))
    print(f"  {'Portfolio (all)':22s} {p5_all:>+7.3f}R")

    for mult in regime_order:
        seg = df[df["risk_multiplier"] == mult]
        lbl = regime_labels[mult]
        if seg.empty:
            continue
        r  = seg["r_multiple"].values
        p5 = float(np.percentile(r, 5))
        p1 = float(np.percentile(r, 1))
        mn = float(np.min(r))
        print(f"  {lbl:<22s} {p5:>+7.3f}R  {p1:>+7.3f}R  {mn:>+7.3f}R")

    # ── Regime mix by asset class ─────────────────────────────────────────────
    print(f"\n  REGIME MIX BY ASSET CLASS  (% of class trades per regime)")
    print(f"  {'Asset Class':<14s} {'Low-vol':>8s} {'Mid-vol':>8s} {'High-vol':>8s} {'Trades':>7s}")
    print(f"  {'─'*50}")
    df_ac = df.copy()
    df_ac["ac"] = df_ac["instrument"].map(ASSET_CLASS_MAP)
    for ac in ASSET_CLASSES:
        seg = df_ac[df_ac["ac"] == ac]
        if seg.empty:
            continue
        n_ac = len(seg)
        lo = 100.0 * (seg["risk_multiplier"] == 1.2).sum() / n_ac
        mi = 100.0 * (seg["risk_multiplier"] == 1.0).sum() / n_ac
        hi = 100.0 * (seg["risk_multiplier"] == 0.6).sum() / n_ac
        print(f"  {ac:<14s} {lo:>7.1f}%  {mi:>7.1f}%  {hi:>7.1f}%  {n_ac:>6d}")


def print_asset_class(trades_df: pd.DataFrame):
    print(f"\n{'─'*W}")
    print(f"  ASSET CLASS BREAKDOWN — 2008–2025")
    print(f"{'─'*W}")
    t2 = trades_df.copy()
    t2["ac"] = t2["instrument"].map(ASSET_CLASS_MAP)
    total_pnl = t2["pnl"].sum()

    print(f"  {'Class':<14s} {'Trades':>6s} {'Expect':>8s} {'AvgHold':>8s} "
          f"{'Win%':>6s} {'TotalPnL':>12s}")
    print(f"  {'─'*62}")
    for ac in ASSET_CLASSES:
        seg = t2[t2["ac"] == ac]
        if seg.empty:
            print(f"  {ac:<14s}  — no trades —")
            continue
        r    = seg["r_multiple"].dropna().values
        pnl  = seg["pnl"].sum()
        pct  = pnl / total_pnl * 100 if total_pnl else 0.0
        exp  = float(np.mean(r)) if len(r) else float("nan")
        hold = float(seg["hold_days"].mean()) if "hold_days" in seg.columns else float("nan")
        wr   = 100.0 * float(np.mean(r > 0)) if len(r) else float("nan")
        print(f"  {ac:<14s}"
              f" {len(seg):>6d}"
              f" {exp:>+7.3f}R"
              f"  {hold:>6.1f}d"
              f" {wr:>5.1f}%"
              f" {pnl:>+11.0f}$  ({pct:>+5.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * W)
    print("  VOL SCALING BACKTEST — Speed-Diff Donchian + ATR Percentile Sizing")
    print(f"  Universe : {len(ALL_NO_FX)} instruments  "
          f"(Rates / Equity / Metals / Energy / Agriculture)")
    print(f"  Sleeves  : B_ME=D50  C_ME=D100 | B_EA=D75  C_EA=D150 | B_R=D150  C_R=D300")
    print(f"  Vol scale: ATR pct <33%→1.2×  |  33–66%→1.0×  |  >66%→0.6×")
    print(f"             252-day rolling window  (6-month warmup for percentile)")
    print(f"  Risk     : {RISK_PER_TRADE*100:.1f}%/tr (base)  "
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
        use_vol_scaling    = True,
    )
    eq     = eng.run()
    trades = eng.get_trades_df()
    metrics = compute_all_metrics(eq, trades, INITIAL_EQUITY)

    print_portfolio_metrics(metrics, eq, "2008–2025  Vol-Scaled Speed-Diff")
    print_comparison(metrics, eq)
    print_asset_class(trades)
    print_regime_diagnostics(trades)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs("output/vol_scaling", exist_ok=True)
    eq.to_csv("output/vol_scaling/equity_curve.csv")
    trades.to_csv("output/vol_scaling/trades.csv", index=False)

    print(f"\n{'─'*W}")
    print(f"  Outputs: output/vol_scaling/  (equity_curve.csv, trades.csv)")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
