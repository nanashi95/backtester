"""
ensemble_backtest.py — Test Plan v2
--------------------------------------
One change at a time. No parameter optimization.

Run 1 — Baseline (Breakout family only, no FX):
  Universe : Equities(5) + Energy(3) + Metals(3) + Agriculture(3) = 14 instruments
  Sleeves  : B = Donchian(50)  |  C = Donchian(100)
  Risk     : 0.7%/trade, 1.5×ATR init, 2×ATR trail, 6R global cap

Run 2 — Add FX Trend Family (combined shared-cap portfolio):
  Breakout universe : same 14 non-FX instruments (Sleeves B + C)
  FX-Trend universe : 6 FX majors (Sleeve FX = EMA50/200 crossover)
  Risk     : same 0.7%, same shared 6R cap

Success criteria (not optimised for):
  MAR > 0.25  |  MaxDD < 30%  |  FX expectancy ≥ 0R  |  flat period < 181 months

Usage:
    python3 ensemble_backtest.py
"""

from __future__ import annotations

import sys, os, io, time
import numpy as np
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))

from strategies.donchian_ensemble import (
    DonchianSleeveB, DonchianSleeveC,
    DonchianSleeveBVolScaled, DonchianSleeveCVolScaled,
    DonchianSleeveBPersist, DonchianSleeveCPersist,
)
from strategies.ema_trend_fx import EMATrendFX
from strategies.tsmom import TSMOMStrategy
from engine.ensemble_engine import EnsemblePortfolioEngine
from metrics.metrics_engine import compute_all_metrics

# ── Universes ──────────────────────────────────────────────────────────────────

NON_FX_INSTRUMENTS = [
    # Equities (5)
    "US500", "US100", "DE30", "JP225", "GB100",
    # Energy (3)
    "USOil", "UKOil", "NATGAS",
    # Metals (3)
    "Gold", "Silver", "Copper",
    # Agriculture (3)
    "WHEAT", "SOYBEAN", "Sugar",
]

FX_INSTRUMENTS = [
    "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "USDNOK",
]

ALL_INSTRUMENTS = NON_FX_INSTRUMENTS + FX_INSTRUMENTS   # 20 total

ASSET_CLASS_MAP = {
    "US500": "Equities", "US100": "Equities", "DE30": "Equities",
    "JP225": "Equities", "GB100": "Equities",
    "USOil": "Energy",   "UKOil": "Energy",   "NATGAS": "Energy",
    "Gold":  "Metals",   "Silver": "Metals",  "Copper": "Metals",
    "WHEAT": "Agriculture", "SOYBEAN": "Agriculture", "Sugar": "Agriculture",
    "EURUSD": "FX", "GBPUSD": "FX", "USDCHF": "FX",
    "AUDUSD": "FX", "USDCAD": "FX", "USDNOK": "FX",
}

ASSET_CLASSES = ["Equities", "FX", "Energy", "Metals", "Agriculture"]

# ── Risk ───────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0

FULL_START = "2008-01-01"
FULL_END   = "2025-12-31"

PERIODS = [
    ("2008-01-01", "2016-12-31", "2008–2016"),
    ("2017-01-01", "2022-12-31", "2017–2022"),
    ("2023-01-01", "2025-12-31", "2023–2025"),
    ("2008-01-01", "2025-12-31", "2008–2025 (Full)"),
]

W = 72


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_period(strategies: dict, start: str, end: str,
               instruments: List[str],
               sleeve_instruments: Optional[Dict[str, List[str]]] = None,
               verbose: bool = True,
               use_pyramiding: bool = False,
               pyramid_min_days: int = 20,
               pyramid_risk_fraction: float = 1.0):
    kwargs = dict(
        strategies             = strategies,
        start                  = start,
        end                    = end,
        initial_equity         = INITIAL_EQUITY,
        instruments            = instruments,
        risk_per_trade         = RISK_PER_TRADE,
        global_cap             = GLOBAL_CAP,
        sleeve_instruments     = sleeve_instruments,
        use_pyramiding         = use_pyramiding,
        pyramid_min_days       = pyramid_min_days,
        pyramid_risk_fraction  = pyramid_risk_fraction,
    )
    if not verbose:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            eng = EnsemblePortfolioEngine(**kwargs)
            eq  = eng.run()
    else:
        eng = EnsemblePortfolioEngine(**kwargs)
        eq  = eng.run()
    trades = eng.get_trades_df()
    return eng, eq, trades


def run_all_periods(strategies: dict, instruments: List[str],
                    sleeve_instruments: Optional[Dict[str, List[str]]] = None,
                    verbose: bool = True,
                    use_pyramiding: bool = False,
                    pyramid_min_days: int = 20,
                    pyramid_risk_fraction: float = 1.0) -> list:
    results = []
    for start, end, label in PERIODS:
        if verbose:
            print(f"\n{'─'*W}\n  Running: {label}\n{'─'*W}")
        eng, eq, trades = run_period(strategies, start, end, instruments,
                                     sleeve_instruments, verbose=verbose,
                                     use_pyramiding=use_pyramiding,
                                     pyramid_min_days=pyramid_min_days,
                                     pyramid_risk_fraction=pyramid_risk_fraction)
        metrics = compute_all_metrics(eq, trades, INITIAL_EQUITY)
        results.append((label, start, end, eng, eq, trades, metrics))
    return results


def run_single_b_reference(start: str, end: str,
                            instruments: List[str]) -> dict:
    from main import run_strategy
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        metrics, *_ = run_strategy(
            DonchianSleeveB(), start, end, INITIAL_EQUITY,
            instruments=instruments, mode="A",
            cap_per_bucket=100.0, cap_total=GLOBAL_CAP, silent=True,
        )
    return metrics


# ── Stats helpers ──────────────────────────────────────────────────────────────

def skew(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 3:
        return float("nan")
    mu, std = np.mean(arr), np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 3) * n * n / ((n-1) * (n-2)))


def _daily_pnl_by_group(trades_df: pd.DataFrame,
                          start: str, end: str,
                          groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Daily closed PnL per group (sleeve-set) as wide DataFrame."""
    dates  = pd.date_range(start, end, freq="D")
    frames = {}
    for gname, sleeves in groups.items():
        df    = trades_df[trades_df["sleeve"].isin(sleeves)]
        if df.empty:
            frames[gname] = pd.Series(0.0, index=dates)
            continue
        daily = df.groupby("exit_date")["pnl"].sum()
        daily.index = pd.to_datetime(daily.index)
        frames[gname] = daily.reindex(dates, fill_value=0.0)
    return pd.DataFrame(frames)


def compute_sleeve_stats(trades_df: pd.DataFrame,
                          start: str, end: str,
                          sleeves: list) -> dict:
    n_years   = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    total_pnl = trades_df["pnl"].sum()
    dp        = _daily_pnl_by_group(trades_df, start, end,
                                    {s: [s] for s in sleeves})
    stats = {}
    for s in sleeves:
        df   = trades_df[trades_df["sleeve"] == s]
        pnl  = df["pnl"].sum()
        pct  = pnl / total_pnl * 100 if total_pnl else 0.0
        eq_e = INITIAL_EQUITY + pnl
        sa_c = (eq_e / INITIAL_EQUITY) ** (1/n_years) - 1 if n_years > 0 else 0.0
        rets = dp[s] / INITIAL_EQUITY
        sh   = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
        stats[s] = {"n_trades": len(df), "total_pnl": pnl,
                    "pct_pnl": pct, "standalone_cagr": sa_c, "sharpe": sh}
    corr = dp.corr()
    return {"sleeves": stats, "correlation": corr}


def compute_family_stats(trades_df: pd.DataFrame,
                          start: str, end: str,
                          families: Dict[str, List[str]]) -> dict:
    """Per-family contribution + cross-family correlation."""
    n_years   = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    total_pnl = trades_df["pnl"].sum()
    dp        = _daily_pnl_by_group(trades_df, start, end, families)

    stats = {}
    for fname, sleeves in families.items():
        df   = trades_df[trades_df["sleeve"].isin(sleeves)]
        pnl  = df["pnl"].sum()
        pct  = pnl / total_pnl * 100 if total_pnl else 0.0
        eq_e = INITIAL_EQUITY + pnl
        sa_c = (eq_e / INITIAL_EQUITY) ** (1/n_years) - 1 if n_years > 0 else 0.0
        rets = dp[fname] / INITIAL_EQUITY
        sh   = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
        r    = df["r_multiple"].values
        wi   = df[df["r_multiple"] > 0]
        lo   = df[df["r_multiple"] <= 0]
        stats[fname] = {
            "n_trades":        len(df),
            "total_pnl":       pnl,
            "pct_pnl":         pct,
            "standalone_cagr": sa_c,
            "sharpe":          sh,
            "expectancy_r":    float(np.mean(r)) if len(r) else float("nan"),
            "win_rate":        100.0 * len(wi) / len(df) if len(df) else 0.0,
            "avg_win_r":       float(wi["r_multiple"].mean()) if len(wi) else float("nan"),
            "avg_loss_r":      float(lo["r_multiple"].mean()) if len(lo) else float("nan"),
            "r_skew":          skew(r) if len(r) >= 3 else float("nan"),
        }
    corr = dp.corr()
    return {"families": stats, "correlation": corr, "names": list(families)}


def compute_asset_class_stats(trades_df: pd.DataFrame) -> dict:
    trades_df = trades_df.copy()
    trades_df["asset_class"] = trades_df["instrument"].map(ASSET_CLASS_MAP)
    total_pnl = trades_df["pnl"].sum()
    stats = {}
    for ac in ASSET_CLASSES:
        df = trades_df[trades_df["asset_class"] == ac]
        if df.empty:
            stats[ac] = None
            continue
        r  = df["r_multiple"].values
        wi = df[df["r_multiple"] > 0]
        lo = df[df["r_multiple"] <= 0]
        stats[ac] = {
            "n_trades":     len(df),
            "pct_pnl":      df["pnl"].sum() / total_pnl * 100 if total_pnl else 0.0,
            "total_pnl":    df["pnl"].sum(),
            "expectancy_r": float(np.mean(r)),
            "avg_hold":     float(df["hold_days"].mean()),
            "win_rate":     100.0 * len(wi) / len(df),
            "avg_win_r":    float(wi["r_multiple"].mean()) if len(wi) else float("nan"),
            "avg_loss_r":   float(lo["r_multiple"].mean()) if len(lo) else float("nan"),
            "r_skew":       skew(r),
        }
    return stats


def compute_vol_scale_stats(trades_df: pd.DataFrame) -> dict:
    """Summarise vol-scaling multiplier distribution across all trades."""
    if "risk_multiplier" not in trades_df.columns or trades_df.empty:
        return {}
    rm = trades_df["risk_multiplier"].values.astype(float)
    n       = len(rm)
    n_up    = int(np.sum(rm > 1.0 + 1e-9))
    n_down  = int(np.sum(rm < 1.0 - 1e-9))
    n_flat  = n - n_up - n_down
    avg_rm  = float(np.mean(rm))
    return {
        "n_trades":          n,
        "avg_risk_mult":     avg_rm,
        "avg_eff_risk_pct":  avg_rm * RISK_PER_TRADE * 100,
        "pct_up":            100.0 * n_up   / n,
        "pct_flat":          100.0 * n_flat / n,
        "pct_down":          100.0 * n_down / n,
    }


def compute_pyramid_stats(trades_df: pd.DataFrame) -> dict:
    """Per-trade pyramid statistics from a completed backtest."""
    if trades_df.empty or "pyramid_added" not in trades_df.columns:
        return {}
    n_total  = len(trades_df)
    n_pyr    = int(trades_df["pyramid_added"].sum())
    r        = trades_df["r_multiple"].dropna().values.astype(float)
    r_pyr    = trades_df[trades_df["pyramid_added"] == True]["r_multiple"].dropna().values
    r_nopyr  = trades_df[trades_df["pyramid_added"] == False]["r_multiple"].dropna().values
    return {
        "n_pyramid":        n_pyr,
        "pct_pyramid":      100.0 * n_pyr / n_total if n_total else 0.0,
        "avg_r_all":        float(np.mean(r))      if len(r)      else float("nan"),
        "r_skew_all":       skew(r)                if len(r) >= 3 else float("nan"),
        "max_r":            float(np.max(r))        if len(r)      else float("nan"),
        "max_pnl":          float(trades_df["pnl"].max()) if n_total else float("nan"),
        "avg_r_pyramid":    float(np.mean(r_pyr))   if len(r_pyr)  else float("nan"),
        "avg_r_nopyramid":  float(np.mean(r_nopyr)) if len(r_nopyr) else float("nan"),
        "r_skew_pyramid":   skew(r_pyr)             if len(r_pyr) >= 3 else float("nan"),
    }


def dominant_sleeve(trades_df: pd.DataFrame, sleeves: list) -> str:
    pnl = {s: trades_df[trades_df["sleeve"] == s]["pnl"].sum() for s in sleeves}
    return max(pnl, key=pnl.get)


def flat_periods(eq_curve: pd.DataFrame, threshold_months: int = 18) -> List[Tuple]:
    eq = eq_curve["total_value"].copy()
    eq.index = pd.to_datetime(eq.index)
    rolling_max = eq.cummax()
    at_high = eq >= rolling_max * 0.9999
    periods, in_uw, t0 = [], False, None
    for date, is_hi in at_high.items():
        if not is_hi and not in_uw:
            in_uw, t0 = True, date
        elif is_hi and in_uw:
            mo = (date - t0).days / 30.44
            if mo >= threshold_months:
                periods.append((t0, date, mo))
            in_uw = False
    if in_uw:
        mo = (eq.index[-1] - t0).days / 30.44
        if mo >= threshold_months:
            periods.append((t0, eq.index[-1], mo))
    return periods


# ── Formatters ─────────────────────────────────────────────────────────────────

def hr(char="─", label=""):
    if label:
        pad = (W - len(label) - 2) // 2
        return f"{'─'*pad} {label} {'─'*(W-pad-len(label)-2)}"
    return char * W


def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v == v else "nan"


# ── Print sections ─────────────────────────────────────────────────────────────

def print_section_a(metrics: dict, label: str):
    print(f"\n{hr('═')}")
    print(f"  A) PORTFOLIO METRICS — {label}")
    print(hr('═'))
    n_yr = metrics.get("n_years", 1)
    tr   = metrics.get("total_trades", 0)
    print(f"  CAGR             : {metrics.get('cagr',0)*100:+.2f}%")
    print(f"  Max Drawdown     : {metrics.get('max_drawdown',0)*100:.2f}%")
    print(f"  MAR Ratio        : {fmt_f(metrics.get('mar_ratio', float('nan')), 3)}")
    print(f"  Sharpe           : {fmt_f(metrics.get('sharpe', float('nan')), 2)}")
    print(f"  Profit Factor    : {fmt_f(metrics.get('profit_factor', float('nan')), 2)}")
    print(f"  Total Trades     : {tr}  ({tr/n_yr if n_yr else 0:.0f}/yr)")
    print(f"  Win Rate         : {metrics.get('win_rate',0)*100:.1f}%")
    print(f"  Expectancy       : {fmt_f(metrics.get('expectancy_r',0), 3)}R")
    print(f"  Longest UW       : {metrics.get('longest_underwater_days',0)} days")


def print_section_b(sleeve_stats: dict, label: str,
                     sleeves: list, sleeve_label: dict):
    print(f"\n{hr('─', f'B) SLEEVE CONTRIBUTION — {label}')}")
    s = sleeve_stats["sleeves"]
    print(f"  {'Sleeve':<16s}  {'Trades':>7s}  {'% PnL':>7s}  {'SA-CAGR':>8s}  {'Sharpe':>7s}")
    print(f"  {'─'*56}")
    for sl in sleeves:
        sd  = s.get(sl, {})
        lbl = sleeve_label.get(sl, sl)
        print(f"  {sl+' ('+lbl+')':16s}"
              f"  {sd.get('n_trades',0):>7d}"
              f"  {sd.get('pct_pnl',0):>6.1f}%"
              f"  {sd.get('standalone_cagr',0)*100:>+7.2f}%"
              f"  {sd.get('sharpe', float('nan')):>7.2f}")
    corr  = sleeve_stats["correlation"]
    pairs = [(a, b) for i, a in enumerate(sleeves) for b in sleeves[i+1:]]
    print(f"\n  Correlation:  " +
          "   ".join(f"{a}↔{b} = {corr.loc[a,b]:+.3f}" for a, b in pairs))


def print_section_c(ac_stats: dict, label: str):
    print(f"\n{hr('─', f'C) ASSET CLASS — {label}')}")
    print(f"  {'Class':<14s} {'Trades':>6s} {'Expect':>7s} {'AvgHold':>8s} "
          f"{'Win%':>6s} {'AvgW':>6s} {'AvgL':>6s} {'Skew':>6s} {'TotalPnL':>10s}")
    print(f"  {'─'*75}")
    for ac in ASSET_CLASSES:
        s = ac_stats.get(ac)
        if s is None:
            print(f"  {ac:<14s}  — no trades —")
            continue
        print(f"  {ac:<14s}"
              f" {s['n_trades']:>6d}"
              f" {s['expectancy_r']:>+6.3f}R"
              f"  {s['avg_hold']:>6.1f}d"
              f" {s['win_rate']:>5.1f}%"
              f" {s['avg_win_r']:>5.2f}R"
              f" {s['avg_loss_r']:>5.2f}R"
              f" {s['r_skew']:>+5.2f}"
              f" {s['total_pnl']:>+10.0f}$")


def print_section_d(regime_rows: list):
    print(f"\n{hr('─', 'D) REGIME DIAGNOSTICS')}")
    print(f"  {'Period':<16s} {'CAGR':>7s} {'MaxDD':>7s} {'MAR':>6s} "
          f"{'Sharpe':>7s} {'Dom.Sleeve':>12s}")
    print(f"  {'─'*62}")
    for lbl, m, dom in regime_rows:
        print(f"  {lbl:<16s} {m.get('cagr',0)*100:>+6.2f}%  "
              f"{m.get('max_drawdown',0)*100:>6.2f}%  "
              f"{fmt_f(m.get('mar_ratio',float('nan')),3):>6s}  "
              f"{fmt_f(m.get('sharpe',float('nan')),2):>6s}  "
              f"Sleeve {dom}")


def print_section_e(diag: dict, single_ref: dict,
                     sleeves: list, sleeve_label: dict):
    print(f"\n{hr('─', 'E) CRITICAL DIAGNOSTICS')}")
    corr  = diag["corr"]
    pairs = [(a, b) for i, a in enumerate(sleeves) for b in sleeves[i+1:]]

    # Q1
    all_below = all(abs(corr.loc[a, b]) < 0.8 for a, b in pairs)
    print(f"\n  1. Sleeve correlation < 0.8?")
    print(f"     " + "  ".join(f"{a}↔{b}={corr.loc[a,b]:+.3f}" for a, b in pairs))
    print(f"     → {'YES ✓' if all_below else 'NO ✗'}")

    # Q2
    pcts  = diag["pct_pnl"]
    max_s, max_v = max(pcts.items(), key=lambda x: x[1])
    print(f"\n  2. Any sleeve >60% of total PnL?")
    for s in sleeves:
        print(f"     {s} ({sleeve_label.get(s,s)}): {pcts[s]:+.1f}%")
    print(f"     → {'YES ✗  ('+max_s+'='+f'{max_v:.1f}%'+')' if max_v > 60 else 'NO ✓'}")

    # Q3
    ens_mar = diag["ensemble_mar"]
    ref_mar = single_ref.get("mar_ratio", float("nan"))
    print(f"\n  3. MAR > single D50 reference?")
    print(f"     Ensemble {fmt_f(ens_mar,3)}  vs  Single {fmt_f(ref_mar,3)}")
    print(f"     → {'YES ✓' if ens_mar > ref_mar else 'NO ✗'}")

    # Q4 — MAR vs success threshold
    print(f"\n  4. MAR > 0.25 (success threshold)?")
    print(f"     → {'YES ✓' if ens_mar > 0.25 else 'NO ✗  ('+fmt_f(ens_mar,3)+')'}")

    # Q5 — MaxDD < 30%
    ens_dd   = diag["ensemble_maxdd"] * 100
    dd_str   = f"{ens_dd:.1f}%"
    print(f"\n  5. MaxDD < 30%?")
    print(f"     → {'YES ✓' if abs(ens_dd) < 30 else 'NO ✗  (' + dd_str + ')'}")

    # Q6 — flat periods
    fps = diag["flat_periods"]
    print(f"\n  6. Flat periods > 18 months?")
    if fps:
        for t0, t1, mo in fps:
            print(f"     {t0.date()} → {t1.date()} : {mo:.1f} months")
        print(f"     → YES ✗")
    else:
        print(f"     → NO ✓")
    print()


def print_section_g(family_stats: dict, label: str):
    """Family breakdown: Breakout vs FX-Trend."""
    print(f"\n{hr('─', f'G) FAMILY BREAKDOWN — {label}')}")
    fs    = family_stats["families"]
    names = family_stats["names"]
    print(f"  {'Family':<18s} {'Trades':>7s} {'% PnL':>7s} {'SA-CAGR':>8s} "
          f"{'Sharpe':>7s} {'Expect':>8s} {'Win%':>6s} {'Skew':>6s}")
    print(f"  {'─'*70}")
    for fname in names:
        fd = fs.get(fname, {})
        print(f"  {fname:<18s}"
              f"  {fd.get('n_trades',0):>6d}"
              f"  {fd.get('pct_pnl',0):>6.1f}%"
              f"  {fd.get('standalone_cagr',0)*100:>+7.2f}%"
              f"  {fd.get('sharpe',float('nan')):>6.2f}"
              f"  {fd.get('expectancy_r',float('nan')):>+7.3f}R"
              f"  {fd.get('win_rate',0):>5.1f}%"
              f"  {fd.get('r_skew',float('nan')):>+5.2f}")
    corr  = family_stats["correlation"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]]
    if pairs:
        parts = "   ".join(f"{a}↔{b} = {corr.loc[a,b]:+.3f}" for a, b in pairs)
        print(f"\n  Family correlation:  {parts}")


def print_asset_by_period(period_results: list):
    print(f"\n{hr('─', 'ASSET CLASS PnL BY PERIOD')}")
    acs_show = [ac for ac in ASSET_CLASSES
                if any(
                    not trades[trades["instrument"].map(ASSET_CLASS_MAP) == ac].empty
                    for _, _, _, _, _, trades, _ in period_results
                )]
    print(f"  {'Period':<16s}  " + "  ".join(f"{ac[:7]:>10s}" for ac in acs_show))
    print(f"  {'─'*72}")
    for label, _, _, _, _, trades, _ in period_results:
        t2  = trades.copy()
        t2["asset_class"] = t2["instrument"].map(ASSET_CLASS_MAP)
        row = f"  {label:<16s}"
        for ac in acs_show:
            p = t2[t2["asset_class"] == ac]["pnl"].sum()
            row += f"  {p:>+9.0f}"
        print(row)


# ── Full config analysis ───────────────────────────────────────────────────────

def analyze_config(period_results: list, ref_metrics: dict,
                   sleeves: list, sleeve_label: dict,
                   config_name: str,
                   families: Optional[Dict[str, List[str]]] = None) -> dict:
    """Print all sections A–G and return summary dict for comparison table."""
    full_label, full_start, full_end, _, full_eq, full_trades, full_metrics = period_results[-1]

    print(f"\n{'═'*W}")
    print(f"  FULL PERIOD ANALYSIS [{config_name}]: {full_label}")
    print(f"{'═'*W}")

    print_section_a(full_metrics, full_label)

    sl_stats = compute_sleeve_stats(full_trades, full_start, full_end, sleeves)
    print_section_b(sl_stats, full_label, sleeves, sleeve_label)

    ac_stats = compute_asset_class_stats(full_trades)
    print_section_c(ac_stats, full_label)

    regime_rows = [(lbl, m, dominant_sleeve(tr, sleeves))
                   for lbl, _, _, _, _, tr, m in period_results]
    print_section_d(regime_rows)

    pct_pnl = {s: sl_stats["sleeves"][s]["pct_pnl"] for s in sleeves}
    fps      = flat_periods(full_eq)
    diag = {
        "corr":           sl_stats["correlation"],
        "pct_pnl":        pct_pnl,
        "ensemble_mar":   full_metrics.get("mar_ratio", float("nan")),
        "ensemble_maxdd": full_metrics.get("max_drawdown", 0),
        "ensemble_cagr":  full_metrics.get("cagr", 0),
        "flat_periods":   fps,
    }
    print_section_e(diag, ref_metrics, sleeves, sleeve_label)

    family_stats = None
    if families:
        family_stats = compute_family_stats(full_trades, full_start, full_end, families)
        print_section_g(family_stats, full_label)

    print_asset_by_period(period_results)

    # Summary for comparison table
    longest_flat = max((mo for _, _, mo in fps), default=0.0)
    fx_ac  = ac_stats.get("FX") or {}
    bk_all = {ac: ac_stats.get(ac) for ac in ["Equities","Energy","Metals","Agriculture"]}
    bk_exp = float(np.mean([s["expectancy_r"] for s in bk_all.values() if s]))

    family_corr = None
    if family_stats and len(family_stats["names"]) >= 2:
        names = family_stats["names"]
        corr  = family_stats["correlation"]
        family_corr = corr.loc[names[0], names[1]]

    return {
        "name":            config_name,
        "total_trades":    full_metrics.get("total_trades", 0),
        "trades_by_sleeve":{s: sl_stats["sleeves"][s]["n_trades"] for s in sleeves},
        "cagr":            full_metrics.get("cagr", 0),
        "maxdd":           full_metrics.get("max_drawdown", 0),
        "mar":             full_metrics.get("mar_ratio", float("nan")),
        "sharpe":          full_metrics.get("sharpe", float("nan")),
        "longest_flat":    longest_flat,
        "fx_expectancy":   fx_ac.get("expectancy_r", float("nan")),
        "fx_trades":       fx_ac.get("n_trades", 0),
        "breakout_exp":    bk_exp,
        "family_corr":     family_corr,
        "sleeves":         sleeves,
        "sleeve_label":    sleeve_label,
        "families":        families,
        "family_stats":    family_stats,
    }


# ── Comparison table ───────────────────────────────────────────────────────────

def print_comparison(summaries: list):
    print(f"\n{'═'*W}")
    print(f"  H) COMPARISON TABLE — 2008–2025 (Full Period)")
    print(f"{'═'*W}")

    col_w = max(len(s["name"]) for s in summaries) + 2

    def row(label, vals):
        print(f"  {label:<30s}" + "".join(f"  {v:>{col_w}s}" for v in vals))

    hdr = [f"[{s['name']}]" for s in summaries]
    print(f"\n  {'Metric':<30s}" + "".join(f"  {h:>{col_w}s}" for h in hdr))
    print(f"  {'─'*(30 + (col_w+2)*len(summaries))}")

    row("Total Trades",      [str(s["total_trades"]) for s in summaries])

    # Sleeve breakdown
    all_sleeves = sorted({sl for s in summaries for sl in s["sleeves"]})
    for sl in all_sleeves:
        vals = []
        for s in summaries:
            if sl in s["sleeves"]:
                n   = s["trades_by_sleeve"].get(sl, 0)
                lbl = s["sleeve_label"].get(sl, sl)
                vals.append(f"{n} ({lbl})")
            else:
                vals.append("—")
        row(f"  Sleeve {sl}", vals)

    row("CAGR",              [f"{s['cagr']*100:+.2f}%" for s in summaries])
    row("Max Drawdown",      [f"{s['maxdd']*100:.2f}%" for s in summaries])
    row("MAR",               [fmt_f(s["mar"], 3) for s in summaries])
    row("Sharpe",            [fmt_f(s["sharpe"], 2) for s in summaries])
    row("Longest Flat (mo)", [f"{s['longest_flat']:.1f}" for s in summaries])
    row("FX Expectancy",
        [f"{s['fx_expectancy']:+.3f}R ({s['fx_trades']}tr)" for s in summaries])
    row("Breakout Avg Exp",
        [f"{s['breakout_exp']:+.3f}R" for s in summaries])

    # Family breakdown (only for configs that have families)
    print(f"\n  {'─'*20}  Family-level (combined config only)  {'─'*10}")
    for s in summaries:
        if not s["families"]:
            continue
        fs = s["family_stats"]["families"]
        for fname, fd in fs.items():
            print(f"  {s['name']} / {fname:<16s}: "
                  f"{fd['n_trades']:>4d}tr  "
                  f"exp={fd['expectancy_r']:>+.3f}R  "
                  f"SA-CAGR={fd['standalone_cagr']*100:>+.2f}%  "
                  f"Sharpe={fd['sharpe']:>.2f}")
        if s["family_corr"] is not None:
            names = s["family_stats"]["names"]
            print(f"  {s['name']} / family corr {names[0]}↔{names[1]}: "
                  f"{s['family_corr']:+.3f}")

    # Success criteria check
    print(f"\n  {'─'*20}  SUCCESS CRITERIA CHECK  {'─'*20}")
    for s in summaries:
        mar_ok  = s["mar"] > 0.25   if s["mar"] == s["mar"] else False
        dd_ok   = abs(s["maxdd"]) < 0.30
        fx_ok   = s["fx_expectancy"] >= 0.0 if s["fx_expectancy"] == s["fx_expectancy"] else False
        flat_ok = s["longest_flat"] < 120.0
        print(f"\n  [{s['name']}]")
        print(f"    MAR > 0.25     : {'✓' if mar_ok  else '✗'}  ({fmt_f(s['mar'],3)})")
        print(f"    MaxDD < 30%    : {'✓' if dd_ok   else '✗'}  ({s['maxdd']*100:.1f}%)")
        print(f"    FX exp ≥ 0R    : {'✓' if fx_ok   else '✗'}  ({s['fx_expectancy']:+.3f}R, {s['fx_trades']}tr)")
        print(f"    Flat < 120 mo  : {'✓' if flat_ok else '✗'}  ({s['longest_flat']:.1f} mo)")
    print()


# ── Vol-scaling comparison table ───────────────────────────────────────────────

def print_vol_comparison(period_results_base: list,
                          period_results_vol:  list,
                          vs_full_base: dict,
                          vs_full_vol:  dict):
    """Side-by-side sub-period + full-period comparison with vol-scaling stats."""
    configs = [
        ("Baseline B+C",         period_results_base, vs_full_base),
        ("Vol-scaled B+C",       period_results_vol,  vs_full_vol),
    ]

    print(f"\n{'═'*W}")
    print(f"  VOL-SCALING COMPARISON — Sub-period & Full Period")
    print(f"{'═'*W}")
    print(f"\n  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'MAR':>6s}  "
          f"{'Sharpe':>6s}  {'Tr/yr':>6s}  {'Config'}")
    print(f"  {'─'*72}")

    for config_name, pr, _ in configs:
        for label, start, end, _, _, _, m in pr:
            n_yr = m.get("n_years", 1)
            tr   = m.get("total_trades", 0)
            print(f"  {label:<16s}  "
                  f"{m.get('cagr',0)*100:>+6.2f}%  "
                  f"{m.get('max_drawdown',0)*100:>6.2f}%  "
                  f"{fmt_f(m.get('mar_ratio', float('nan')), 3):>6s}  "
                  f"{fmt_f(m.get('sharpe', float('nan')), 2):>6s}  "
                  f"{tr/n_yr if n_yr else 0:>5.0f}/y  "
                  f"[{config_name}]")
        print(f"  {'─'*72}")

    # Full-period head-to-head
    print(f"\n{'─'*W}")
    print(f"  FULL PERIOD HEAD-TO-HEAD (2008–2025)")
    print(f"{'─'*W}")
    col_w = 20
    def row(label, a, b):
        print(f"  {label:<28s}  {a:>{col_w}s}  {b:>{col_w}s}")

    row("Metric", "[Baseline B+C]", "[Vol-scaled B+C]")
    print(f"  {'─'*72}")

    base_m = period_results_base[-1][6]
    vol_m  = period_results_vol[-1][6]
    _, fs, fe = period_results_base[-1][1], period_results_base[-1][1], period_results_base[-1][2]
    fps_base   = flat_periods(period_results_base[-1][4])
    fps_vol    = flat_periods(period_results_vol[-1][4])
    flat_base  = max((mo for _, _, mo in fps_base), default=0.0)
    flat_vol   = max((mo for _, _, mo in fps_vol),  default=0.0)
    n_yr       = base_m.get("n_years", 1)

    row("CAGR",
        f"{base_m.get('cagr',0)*100:+.2f}%",
        f"{vol_m.get('cagr',0)*100:+.2f}%")
    row("Max Drawdown",
        f"{base_m.get('max_drawdown',0)*100:.2f}%",
        f"{vol_m.get('max_drawdown',0)*100:.2f}%")
    row("MAR",
        fmt_f(base_m.get("mar_ratio", float("nan")), 3),
        fmt_f(vol_m.get("mar_ratio", float("nan")), 3))
    row("Sharpe",
        fmt_f(base_m.get("sharpe", float("nan")), 2),
        fmt_f(vol_m.get("sharpe", float("nan")), 2))
    row("Profit Factor",
        fmt_f(base_m.get("profit_factor", float("nan")), 2),
        fmt_f(vol_m.get("profit_factor", float("nan")), 2))
    row("Trades/yr",
        f"{base_m.get('total_trades',0)/n_yr:.0f}",
        f"{vol_m.get('total_trades',0)/n_yr:.0f}")
    row("Longest flat (mo)",
        f"{flat_base:.1f}",
        f"{flat_vol:.1f}")

    # Vol-scaling stats
    tr_vol = period_results_vol[-1][5]
    vs     = compute_vol_scale_stats(tr_vol)
    tr_base = period_results_base[-1][5]
    vsb     = compute_vol_scale_stats(tr_base)

    print(f"\n  {'─'*36}  Scaling stats  {'─'*18}")
    row("Avg effective risk/trade",
        f"{RISK_PER_TRADE*100:.2f}%  (fixed)",
        f"{vs.get('avg_eff_risk_pct', RISK_PER_TRADE*100):.3f}%")
    row("% trades scaled up (1.2×)",
        "—",
        f"{vs.get('pct_up', 0):.1f}%")
    row("% trades at base (1.0×)",
        "—",
        f"{vs.get('pct_flat', 0):.1f}%")
    row("% trades scaled down (0.6×)",
        "—",
        f"{vs.get('pct_down', 0):.1f}%")

    print()


# ── Pyramid comparison table ───────────────────────────────────────────────────

def print_pyramid_comparison(period_results_base: list,
                              period_results_pyr:  list):
    """Side-by-side sub-period + full-period comparison for pyramid test."""
    configs = [
        ("Baseline B+C",  period_results_base),
        ("Pyramid B+C",   period_results_pyr),
    ]

    print(f"\n{'═'*W}")
    print(f"  PYRAMID COMPARISON — Sub-period & Full Period")
    print(f"{'═'*W}")
    print(f"\n  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'MAR':>6s}  "
          f"{'Sharpe':>6s}  {'Tr/yr':>5s}  {'Config'}")
    print(f"  {'─'*72}")

    for config_name, pr in configs:
        for label, start, end, _, _, _, m in pr:
            n_yr = m.get("n_years", 1)
            tr   = m.get("total_trades", 0)
            print(f"  {label:<16s}  "
                  f"{m.get('cagr',0)*100:>+6.2f}%  "
                  f"{m.get('max_drawdown',0)*100:>6.2f}%  "
                  f"{fmt_f(m.get('mar_ratio', float('nan')), 3):>6s}  "
                  f"{fmt_f(m.get('sharpe', float('nan')), 2):>6s}  "
                  f"{tr/n_yr if n_yr else 0:>4.0f}/y  "
                  f"[{config_name}]")
        print(f"  {'─'*72}")

    # Full-period head-to-head
    base_m = period_results_base[-1][6]
    pyr_m  = period_results_pyr[-1][6]
    tr_base = period_results_base[-1][5]
    tr_pyr  = period_results_pyr[-1][5]
    fps_base = flat_periods(period_results_base[-1][4])
    fps_pyr  = flat_periods(period_results_pyr[-1][4])
    flat_base = max((mo for _, _, mo in fps_base), default=0.0)
    flat_pyr  = max((mo for _, _, mo in fps_pyr),  default=0.0)
    n_yr      = base_m.get("n_years", 1)

    ps_base = compute_pyramid_stats(tr_base)
    ps_pyr  = compute_pyramid_stats(tr_pyr)

    r_base = tr_base["r_multiple"].dropna().values.astype(float)
    r_pyr  = tr_pyr["r_multiple"].dropna().values.astype(float)

    print(f"\n{'─'*W}")
    print(f"  FULL PERIOD HEAD-TO-HEAD (2008–2025)")
    print(f"{'─'*W}")
    col_w = 20

    def row(label, a, b):
        print(f"  {label:<30s}  {a:>{col_w}s}  {b:>{col_w}s}")

    row("Metric", "[Baseline B+C]", "[Pyramid B+C]")
    print(f"  {'─'*76}")
    row("CAGR",
        f"{base_m.get('cagr',0)*100:+.2f}%",
        f"{pyr_m.get('cagr',0)*100:+.2f}%")
    row("Max Drawdown",
        f"{base_m.get('max_drawdown',0)*100:.2f}%",
        f"{pyr_m.get('max_drawdown',0)*100:.2f}%")
    row("MAR",
        fmt_f(base_m.get("mar_ratio", float("nan")), 3),
        fmt_f(pyr_m.get("mar_ratio", float("nan")), 3))
    row("Sharpe",
        fmt_f(base_m.get("sharpe", float("nan")), 2),
        fmt_f(pyr_m.get("sharpe", float("nan")), 2))
    row("Profit Factor",
        fmt_f(base_m.get("profit_factor", float("nan")), 2),
        fmt_f(pyr_m.get("profit_factor", float("nan")), 2))
    row("Trades/yr",
        f"{base_m.get('total_trades',0)/n_yr:.0f}",
        f"{pyr_m.get('total_trades',0)/n_yr:.0f}")
    row("Longest flat (mo)",
        f"{flat_base:.1f}",
        f"{flat_pyr:.1f}")
    row("Avg R/trade",
        fmt_f(ps_base.get("avg_r_all", float("nan")), 3) + "R",
        fmt_f(ps_pyr.get("avg_r_all", float("nan")), 3) + "R")

    print(f"\n  {'─'*30}  Pyramid stats  {'─'*22}")
    row("% trades that got pyramid",
        "—",
        f"{ps_pyr.get('pct_pyramid', 0):.1f}%  "
        f"({ps_pyr.get('n_pyramid', 0)} of {len(tr_pyr)})")

    print(f"\n  {'─'*30}  R-multiple distribution  {'─'*13}")
    row("Avg R (all trades)",
        fmt_f(ps_base.get("avg_r_all", float("nan")), 3) + "R",
        fmt_f(ps_pyr.get("avg_r_all", float("nan")), 3) + "R")
    row("R skew (all trades)",
        fmt_f(skew(r_base) if len(r_base) >= 3 else float("nan"), 2),
        fmt_f(skew(r_pyr)  if len(r_pyr)  >= 3 else float("nan"), 2))
    row("Max single-trade R",
        f"{ps_base.get('max_r', float('nan')):+.2f}R",
        f"{ps_pyr.get('max_r', float('nan')):+.2f}R")
    row("Max single-trade PnL ($)",
        f"${ps_base.get('max_pnl', float('nan')):,.0f}",
        f"${ps_pyr.get('max_pnl', float('nan')):,.0f}")

    if ps_pyr.get("n_pyramid", 0) > 0:
        print(f"\n  {'─'*30}  Pyramided vs non-pyramided (pyramid run)  {'─'*3}")
        row("Avg R — pyramided trades",
            "—",
            fmt_f(ps_pyr.get("avg_r_pyramid", float("nan")), 3) + "R")
        row("Avg R — non-pyramided trades",
            "—",
            fmt_f(ps_pyr.get("avg_r_nopyramid", float("nan")), 3) + "R")
        row("R skew — pyramided trades",
            "—",
            fmt_f(ps_pyr.get("r_skew_pyramid", float("nan")), 2))
    print()


# ── Persistence-filter comparison table ────────────────────────────────────────

def print_persistence_comparison(period_results_base: list,
                                  period_results_prs:  list):
    """Side-by-side comparison: Baseline B+C vs Persistence-filtered B+C."""
    configs = [
        ("Baseline B+C",    period_results_base),
        ("Persist B+C",     period_results_prs),
    ]

    print(f"\n{'═'*W}")
    print(f"  PERSISTENCE FILTER COMPARISON — Sub-period & Full Period")
    print(f"{'═'*W}")
    print(f"\n  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'MAR':>6s}  "
          f"{'Sharpe':>6s}  {'Tr/yr':>5s}  {'WinRate':>7s}  {'Config'}")
    print(f"  {'─'*80}")

    for config_name, pr in configs:
        for label, start, end, _, _, _, m in pr:
            n_yr = m.get("n_years", 1)
            tr   = m.get("total_trades", 0)
            wr   = m.get("win_rate", 0) * 100
            print(f"  {label:<16s}  "
                  f"{m.get('cagr',0)*100:>+6.2f}%  "
                  f"{m.get('max_drawdown',0)*100:>6.2f}%  "
                  f"{fmt_f(m.get('mar_ratio', float('nan')), 3):>6s}  "
                  f"{fmt_f(m.get('sharpe', float('nan')), 2):>6s}  "
                  f"{tr/n_yr if n_yr else 0:>4.0f}/y  "
                  f"{wr:>6.1f}%  "
                  f"[{config_name}]")
        print(f"  {'─'*80}")

    # Full-period head-to-head
    base_m  = period_results_base[-1][6]
    prs_m   = period_results_prs[-1][6]
    tr_base = period_results_base[-1][5]
    tr_prs  = period_results_prs[-1][5]
    fps_base = flat_periods(period_results_base[-1][4])
    fps_prs  = flat_periods(period_results_prs[-1][4])
    flat_base = max((mo for _, _, mo in fps_base), default=0.0)
    flat_prs  = max((mo for _, _, mo in fps_prs),  default=0.0)
    n_yr      = base_m.get("n_years", 1)

    r_base = tr_base["r_multiple"].dropna().values.astype(float)
    r_prs  = tr_prs["r_multiple"].dropna().values.astype(float)

    n_base = len(tr_base)
    n_prs  = len(tr_prs)
    pct_reduction = 100.0 * (n_base - n_prs) / n_base if n_base > 0 else 0.0

    avg_r_base = float(np.mean(r_base)) if len(r_base) else float("nan")
    avg_r_prs  = float(np.mean(r_prs))  if len(r_prs)  else float("nan")
    delta_r    = avg_r_prs - avg_r_base

    print(f"\n{'─'*W}")
    print(f"  FULL PERIOD HEAD-TO-HEAD (2008–2025)")
    print(f"{'─'*W}")
    col_w = 20

    def row(label, a, b):
        print(f"  {label:<30s}  {a:>{col_w}s}  {b:>{col_w}s}")

    row("Metric", "[Baseline B+C]", "[Persist B+C]")
    print(f"  {'─'*76}")
    row("CAGR",
        f"{base_m.get('cagr',0)*100:+.2f}%",
        f"{prs_m.get('cagr',0)*100:+.2f}%")
    row("Max Drawdown",
        f"{base_m.get('max_drawdown',0)*100:.2f}%",
        f"{prs_m.get('max_drawdown',0)*100:.2f}%")
    row("MAR",
        fmt_f(base_m.get("mar_ratio", float("nan")), 3),
        fmt_f(prs_m.get("mar_ratio", float("nan")), 3))
    row("Sharpe",
        fmt_f(base_m.get("sharpe", float("nan")), 2),
        fmt_f(prs_m.get("sharpe", float("nan")), 2))
    row("Profit Factor",
        fmt_f(base_m.get("profit_factor", float("nan")), 2),
        fmt_f(prs_m.get("profit_factor", float("nan")), 2))
    row("Trades/yr",
        f"{base_m.get('total_trades',0)/n_yr:.0f}",
        f"{prs_m.get('total_trades',0)/n_yr:.0f}")
    row("Win Rate",
        f"{base_m.get('win_rate',0)*100:.1f}%",
        f"{prs_m.get('win_rate',0)*100:.1f}%")
    row("Avg R/trade",
        f"{avg_r_base:+.3f}R",
        f"{avg_r_prs:+.3f}R")
    row("R skew",
        fmt_f(skew(r_base) if len(r_base) >= 3 else float("nan"), 2),
        fmt_f(skew(r_prs)  if len(r_prs)  >= 3 else float("nan"), 2))
    row("Longest flat (mo)",
        f"{flat_base:.1f}",
        f"{flat_prs:.1f}")

    print(f"\n  {'─'*30}  Filter impact  {'─'*24}")
    row("Total trades (full period)",
        f"{n_base}",
        f"{n_prs}  ({pct_reduction:+.1f}% vs baseline)")
    row("Δ Avg R/trade vs baseline",
        "—",
        f"{delta_r:+.3f}R  ({'↑' if delta_r > 0 else '↓'})")

    # Asset-class expectancy delta
    ac_base = compute_asset_class_stats(tr_base)
    ac_prs  = compute_asset_class_stats(tr_prs)
    print(f"\n  {'─'*30}  Asset class expectancy  {'─'*16}")
    print(f"  {'Class':<14s}  {'Baseline Exp':>14s}  {'Persist Exp':>13s}  "
          f"{'Δ Exp':>7s}  {'Base Tr':>7s}  {'Prs Tr':>7s}")
    print(f"  {'─'*72}")
    for ac in ["Equities", "Energy", "Metals", "Agriculture"]:
        sb  = ac_base.get(ac) or {}
        sp  = ac_prs.get(ac)  or {}
        eb  = sb.get("expectancy_r", float("nan"))
        ep  = sp.get("expectancy_r", float("nan"))
        dlt = ep - eb if (eb == eb and ep == ep) else float("nan")
        nb  = sb.get("n_trades", 0)
        np_ = sp.get("n_trades", 0)
        print(f"  {ac:<14s}  {eb:>+12.3f}R  {ep:>+11.3f}R  "
              f"{dlt:>+6.3f}R  {nb:>7d}  {np_:>7d}")
    print()


# ── Generic two-engine comparison table ────────────────────────────────────────

def print_engine_comparison(period_results_a: list, label_a: str,
                             period_results_b: list, label_b: str):
    """Side-by-side comparison for any two engine configs."""
    configs = [(label_a, period_results_a), (label_b, period_results_b)]

    print(f"\n{'═'*W}")
    print(f"  COMPARISON — {label_a}  vs  {label_b}")
    print(f"{'═'*W}")
    print(f"\n  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'MAR':>6s}  "
          f"{'Sharpe':>6s}  {'Tr/yr':>5s}  {'WinRate':>7s}  {'Config'}")
    print(f"  {'─'*80}")

    for config_name, pr in configs:
        for label, start, end, _, _, _, m in pr:
            n_yr = m.get("n_years", 1)
            tr   = m.get("total_trades", 0)
            wr   = m.get("win_rate", 0) * 100
            print(f"  {label:<16s}  "
                  f"{m.get('cagr',0)*100:>+6.2f}%  "
                  f"{m.get('max_drawdown',0)*100:>6.2f}%  "
                  f"{fmt_f(m.get('mar_ratio', float('nan')), 3):>6s}  "
                  f"{fmt_f(m.get('sharpe', float('nan')), 2):>6s}  "
                  f"{tr/n_yr if n_yr else 0:>4.0f}/y  "
                  f"{wr:>6.1f}%  "
                  f"[{config_name}]")
        print(f"  {'─'*80}")

    # Full-period head-to-head
    ma  = period_results_a[-1][6]
    mb  = period_results_b[-1][6]
    tra = period_results_a[-1][5]
    trb = period_results_b[-1][5]
    fps_a = flat_periods(period_results_a[-1][4])
    fps_b = flat_periods(period_results_b[-1][4])
    flat_a = max((mo for _, _, mo in fps_a), default=0.0)
    flat_b = max((mo for _, _, mo in fps_b), default=0.0)
    n_yr   = ma.get("n_years", 1)

    ra = tra["r_multiple"].dropna().values.astype(float)
    rb = trb["r_multiple"].dropna().values.astype(float)

    print(f"\n{'─'*W}")
    print(f"  FULL PERIOD HEAD-TO-HEAD (2008–2025)")
    print(f"{'─'*W}")
    col_w = 20

    def row(lbl, a, b):
        print(f"  {lbl:<30s}  {a:>{col_w}s}  {b:>{col_w}s}")

    row("Metric", f"[{label_a}]", f"[{label_b}]")
    print(f"  {'─'*76}")
    row("CAGR",
        f"{ma.get('cagr',0)*100:+.2f}%",
        f"{mb.get('cagr',0)*100:+.2f}%")
    row("Max Drawdown",
        f"{ma.get('max_drawdown',0)*100:.2f}%",
        f"{mb.get('max_drawdown',0)*100:.2f}%")
    row("MAR",
        fmt_f(ma.get("mar_ratio", float("nan")), 3),
        fmt_f(mb.get("mar_ratio", float("nan")), 3))
    row("Sharpe",
        fmt_f(ma.get("sharpe", float("nan")), 2),
        fmt_f(mb.get("sharpe", float("nan")), 2))
    row("Profit Factor",
        fmt_f(ma.get("profit_factor", float("nan")), 2),
        fmt_f(mb.get("profit_factor", float("nan")), 2))
    row("Trades/yr",
        f"{ma.get('total_trades',0)/n_yr:.0f}",
        f"{mb.get('total_trades',0)/n_yr:.0f}")
    row("Win Rate",
        f"{ma.get('win_rate',0)*100:.1f}%",
        f"{mb.get('win_rate',0)*100:.1f}%")
    row("Avg R/trade",
        f"{float(np.mean(ra)) if len(ra) else float('nan'):+.3f}R",
        f"{float(np.mean(rb)) if len(rb) else float('nan'):+.3f}R")
    row("R skew",
        fmt_f(skew(ra) if len(ra) >= 3 else float("nan"), 2),
        fmt_f(skew(rb) if len(rb) >= 3 else float("nan"), 2))
    row("Longest flat (mo)",
        f"{flat_a:.1f}",
        f"{flat_b:.1f}")

    # Asset class
    aca = compute_asset_class_stats(tra)
    acb = compute_asset_class_stats(trb)
    print(f"\n  {'─'*30}  Asset class expectancy  {'─'*16}")
    print(f"  {'Class':<14s}  {f'[{label_a}]':>16s}  {f'[{label_b}]':>14s}  "
          f"{'Δ':>7s}  {'Tr-A':>6s}  {'Tr-B':>6s}")
    print(f"  {'─'*72}")
    for ac in ["Equities", "Energy", "Metals", "Agriculture"]:
        sa  = aca.get(ac) or {}
        sb  = acb.get(ac) or {}
        ea  = sa.get("expectancy_r", float("nan"))
        eb  = sb.get("expectancy_r", float("nan"))
        dlt = eb - ea if (ea == ea and eb == eb) else float("nan")
        na  = sa.get("n_trades", 0)
        nb_ = sb.get("n_trades", 0)
        print(f"  {ac:<14s}  {ea:>+14.3f}R  {eb:>+12.3f}R  "
              f"{dlt:>+6.3f}R  {na:>6d}  {nb_:>6d}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # BASELINE: Breakout B+C — no filter
    # ─────────────────────────────────────────────────────────────────────────
    strategies_base = {"B": DonchianSleeveB(), "C": DonchianSleeveC()}

    print("=" * W)
    print("  BASELINE: Breakout B+C — no entry filter")
    print(f"  Universe: {len(NON_FX_INSTRUMENTS)} instruments  "
          f"(Equities/Energy/Metals/Agriculture)")
    print(f"  Risk:     {RISK_PER_TRADE*100:.1f}%/tr  1.5×ATR init  2×ATR trail  "
          f"{GLOBAL_CAP:.0f}R cap")
    print("=" * W)

    pr_base = run_all_periods(strategies_base, NON_FX_INSTRUMENTS, verbose=True)

    print(f"\n{'─'*W}\n  Running reference: Single D50 on same 14-instr universe")
    ref_base = run_single_b_reference(FULL_START, FULL_END, NON_FX_INSTRUMENTS)

    summary_base = analyze_config(
        pr_base, ref_base, ["B", "C"], {"B": "D50", "C": "D100"}, "Baseline B+C")

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 2: TSMOM(50/100) — Transitional Momentum
    #   Entry : 50-day AND 100-day return aligned → fire on transition day
    #   Exit  : either return flips sign  OR  ATR trailing stop
    # ─────────────────────────────────────────────────────────────────────────
    strategies_tsmom = {"TSMOM": TSMOMStrategy()}

    print(f"\n{'='*W}")
    print("  ENGINE 2 — TSMOM(50/100): Time-Series Momentum")
    print(f"  Long : ret50 > 0  AND  ret100 > 0  (transition day)")
    print(f"  Short: ret50 < 0  AND  ret100 < 0  (transition day)")
    print(f"  Exit : either return flips sign  OR  2×ATR trailing stop")
    print(f"  Risk : {RISK_PER_TRADE*100:.1f}%/tr  1.5×ATR init  {GLOBAL_CAP:.0f}R cap  "
          f"same universe")
    print("=" * W)

    pr_tsmom = run_all_periods(strategies_tsmom, NON_FX_INSTRUMENTS, verbose=True)

    summary_tsmom = analyze_config(
        pr_tsmom, ref_base,
        ["TSMOM"], {"TSMOM": "ret50/100"}, "TSMOM")

    # ─────────────────────────────────────────────────────────────────────────
    # COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    print_engine_comparison(pr_base, "Baseline B+C", pr_tsmom, "TSMOM(50/100)")

    # Save outputs
    os.makedirs("output/ensemble", exist_ok=True)
    _, _, _, _, eq_b, tr_b, _ = pr_base[-1]
    _, _, _, _, eq_t, tr_t, _ = pr_tsmom[-1]
    eq_b.to_csv("output/ensemble/equity_baseline.csv")
    tr_b.to_csv("output/ensemble/trades_baseline.csv", index=False)
    eq_t.to_csv("output/ensemble/equity_tsmom.csv")
    tr_t.to_csv("output/ensemble/trades_tsmom.csv", index=False)

    print(f"{'─'*W}")
    print(f"  Outputs: output/ensemble/  "
          f"(equity_baseline, trades_baseline, equity_tsmom, trades_tsmom)")
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
