"""
ibkr_backtest.py
----------------
Donchian ensemble backtest on the IBKR futures universe.

Wires up the IBKR data loader to the existing ensemble engine — no strategy
or engine code is modified.  Results are directly comparable to
ensemble_backtest.py once IBKR CSVs are downloaded.

Universe (22 instruments):
  Rates        (3): ZN, ZB, ZF
  Equity       (4): ES, NQ, RTY, YM
  FX           (5): 6J, 6E, 6A, 6C, 6B
  Metals       (5): GC, SI, HG, PA, PL
  Energy       (2): CL, NG
  Agriculture  (6): ZW, ZS, ZC, SB, KC, CC

Sleeves: B = Donchian(50)  |  C = Donchian(100)
Risk:    0.7%/trade, 1.5×ATR init, 2×ATR trail, 6R global cap

Usage:
    python3 ibkr_backtest.py

Prerequisites:
    data/ibkr/*.csv files — run ibkr/downloader.py first.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ── Wire up data source BEFORE importing any engine ──────────────────────────
# Historical data: Yahoo Finance futures (2000–today, actual CME/CBOT prices)
# Live execution:  IB Gateway (same underlying contracts — prices match)
import data.loader as loader
import data.yf_data_loader as yf_loader
loader.configure(yf_loader.load_all_data, yf_loader.get_instrument_bucket)

from strategies.donchian_ensemble import DonchianSleeveB, DonchianSleeveC
from engine.ensemble_engine import EnsemblePortfolioEngine
from metrics.metrics_engine import compute_all_metrics

# ── Universe ──────────────────────────────────────────────────────────────────

RATES_INSTRUMENTS = ["ZN", "ZB", "ZF"]
EQUITY_INSTRUMENTS = ["ES", "NQ", "RTY", "YM"]
FX_INSTRUMENTS = ["6J", "6E", "6A", "6C", "6B"]
METALS_INSTRUMENTS = ["GC", "SI", "HG", "PA", "PL"]
ENERGY_INSTRUMENTS = ["CL", "NG"]
AGRICULTURE_INSTRUMENTS = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

ALL_INSTRUMENTS = (
    RATES_INSTRUMENTS
    + EQUITY_INSTRUMENTS
    + FX_INSTRUMENTS
    + METALS_INSTRUMENTS
    + ENERGY_INSTRUMENTS
    + AGRICULTURE_INSTRUMENTS
)

ASSET_CLASS_MAP = {}
for _instr in RATES_INSTRUMENTS:       ASSET_CLASS_MAP[_instr] = "Rates"
for _instr in EQUITY_INSTRUMENTS:      ASSET_CLASS_MAP[_instr] = "Equity"
for _instr in FX_INSTRUMENTS:          ASSET_CLASS_MAP[_instr] = "FX"
for _instr in METALS_INSTRUMENTS:      ASSET_CLASS_MAP[_instr] = "Metals"
for _instr in ENERGY_INSTRUMENTS:      ASSET_CLASS_MAP[_instr] = "Energy"
for _instr in AGRICULTURE_INSTRUMENTS: ASSET_CLASS_MAP[_instr] = "Agriculture"

ASSET_CLASSES = ["Rates", "Equity", "FX", "Metals", "Energy", "Agriculture"]

# ── Risk ──────────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0

PERIODS = [
    ("2008-01-01", "2016-12-31", "2008–2016"),
    ("2017-01-01", "2022-12-31", "2017–2022"),
    ("2023-01-01", "2025-12-31", "2023–2025"),
    ("2008-01-01", "2025-12-31", "2008–2025 (Full)"),
]

W = 72


# ── Runner ────────────────────────────────────────────────────────────────────

def run_period(strategies: dict, start: str, end: str,
               instruments: list[str]) -> tuple:
    eng = EnsemblePortfolioEngine(
        strategies    = strategies,
        start         = start,
        end           = end,
        initial_equity = INITIAL_EQUITY,
        instruments   = instruments,
        risk_per_trade = RISK_PER_TRADE,
        global_cap    = GLOBAL_CAP,
    )
    eq     = eng.run()
    trades = eng.get_trades_df()
    return eng, eq, trades


def run_all_periods(strategies: dict,
                    instruments: list[str]) -> list:
    results = []
    for start, end, label in PERIODS:
        print(f"\n{'─'*W}\n  Running: {label}\n{'─'*W}")
        eng, eq, trades = run_period(strategies, start, end, instruments)
        metrics = compute_all_metrics(eq, trades, INITIAL_EQUITY)
        results.append((label, start, end, eng, eq, trades, metrics))
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def skew(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 3:
        return float("nan")
    mu, std = np.mean(arr), np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mu) / std) ** 3) * n * n / ((n - 1) * (n - 2)))


def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v == v else "nan"


def hr(label=""):
    if label:
        pad = (W - len(label) - 2) // 2
        return f"{'─'*pad} {label} {'─'*(W - pad - len(label) - 2)}"
    return "─" * W


# ── Section printers ─────────────────────────────────────────────────────────

def print_portfolio_metrics(metrics: dict, label: str):
    n_yr = metrics.get("n_years", 1)
    tr   = metrics.get("total_trades", 0)
    print(f"\n{'═'*W}")
    print(f"  PORTFOLIO METRICS — {label}")
    print(f"{'═'*W}")
    print(f"  CAGR             : {metrics.get('cagr', 0)*100:+.2f}%")
    print(f"  Max Drawdown     : {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"  MAR Ratio        : {fmt_f(metrics.get('mar_ratio', float('nan')), 3)}")
    print(f"  Sharpe           : {fmt_f(metrics.get('sharpe', float('nan')), 2)}")
    print(f"  Profit Factor    : {fmt_f(metrics.get('profit_factor', float('nan')), 2)}")
    print(f"  Total Trades     : {tr}  ({tr/n_yr if n_yr else 0:.0f}/yr)")
    print(f"  Win Rate         : {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Expectancy       : {fmt_f(metrics.get('expectancy_r', 0), 3)}R")
    print(f"  Longest UW       : {metrics.get('longest_underwater_days', 0)} days")


def print_regime_table(period_results: list):
    print(f"\n{hr('REGIME DIAGNOSTICS')}")
    print(f"  {'Period':<16s}  {'CAGR':>7s}  {'MaxDD':>7s}  {'MAR':>6s}  "
          f"{'Sharpe':>6s}  {'Trades/yr':>10s}")
    print(f"  {'─'*62}")
    for label, start, end, _, _, _, m in period_results:
        n_yr = m.get("n_years", 1)
        tr   = m.get("total_trades", 0)
        print(f"  {label:<16s}  "
              f"{m.get('cagr', 0)*100:>+6.2f}%  "
              f"{m.get('max_drawdown', 0)*100:>6.2f}%  "
              f"{fmt_f(m.get('mar_ratio', float('nan')), 3):>6s}  "
              f"{fmt_f(m.get('sharpe', float('nan')), 2):>6s}  "
              f"{tr/n_yr if n_yr else 0:>8.0f}/yr")


def print_asset_class(trades_df: pd.DataFrame, label: str):
    print(f"\n{hr(f'ASSET CLASS — {label}')}")
    trades_df = trades_df.copy()
    trades_df["ac"] = trades_df["instrument"].map(ASSET_CLASS_MAP)
    total_pnl = trades_df["pnl"].sum()

    print(f"  {'Class':<14s} {'Trades':>6s} {'Expect':>8s} {'AvgHold':>8s} "
          f"{'Win%':>6s} {'TotalPnL':>12s}")
    print(f"  {'─'*62}")

    for ac in ASSET_CLASSES:
        df = trades_df[trades_df["ac"] == ac]
        if df.empty:
            print(f"  {ac:<14s}  — no trades —")
            continue
        r    = df["r_multiple"].dropna().values
        wi   = df[df["r_multiple"] > 0]
        pnl  = df["pnl"].sum()
        pct  = pnl / total_pnl * 100 if total_pnl else 0.0
        exp  = float(np.mean(r)) if len(r) else float("nan")
        hold = float(df["hold_days"].mean()) if "hold_days" in df.columns else float("nan")
        wr   = 100.0 * len(wi) / len(df)
        print(f"  {ac:<14s}"
              f" {len(df):>6d}"
              f" {exp:>+7.3f}R"
              f"  {hold:>6.1f}d"
              f" {wr:>5.1f}%"
              f" {pnl:>+11.0f}$  ({pct:>+5.1f}%)")


def print_asset_class_by_period(period_results: list):
    print(f"\n{hr('ASSET CLASS PnL BY PERIOD')}")
    print(f"  {'Period':<16s}  " +
          "  ".join(f"{ac[:6]:>9s}" for ac in ASSET_CLASSES))
    print(f"  {'─'*72}")
    for label, _, _, _, _, trades, _ in period_results:
        t2 = trades.copy()
        t2["ac"] = t2["instrument"].map(ASSET_CLASS_MAP)
        row = f"  {label:<16s}"
        for ac in ASSET_CLASSES:
            p = t2[t2["ac"] == ac]["pnl"].sum()
            row += f"  {p:>+8.0f}"
        print(row)


def print_sleeve_stats(trades_df: pd.DataFrame, start: str, end: str):
    sleeves    = ["B", "C"]
    n_years    = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    total_pnl  = trades_df["pnl"].sum()
    dates      = pd.date_range(start, end, freq="D")
    daily_pnl  = {}
    for s in sleeves:
        df   = trades_df[trades_df["sleeve"] == s]
        d    = df.groupby("exit_date")["pnl"].sum()
        d.index = pd.to_datetime(d.index)
        daily_pnl[s] = d.reindex(dates, fill_value=0.0)

    dp = pd.DataFrame(daily_pnl)
    corr = dp.corr()

    print(f"\n{hr('SLEEVE CONTRIBUTION')}")
    print(f"  {'Sleeve':<10s} {'Trades':>7s} {'% PnL':>7s} {'SA-CAGR':>9s} {'Sharpe':>7s}")
    print(f"  {'─'*46}")
    sleeve_labels = {"B": "D50", "C": "D100"}
    for s in sleeves:
        df  = trades_df[trades_df["sleeve"] == s]
        pnl = df["pnl"].sum()
        pct = pnl / total_pnl * 100 if total_pnl else 0.0
        eq_e = INITIAL_EQUITY + pnl
        sacg = (eq_e / INITIAL_EQUITY) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        rets = dp[s] / INITIAL_EQUITY
        sh   = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
        print(f"  {s+' ('+sleeve_labels[s]+')':10s}"
              f" {len(df):>7d}"
              f" {pct:>6.1f}%"
              f" {sacg*100:>+8.2f}%"
              f" {sh:>7.2f}")

    print(f"\n  Sleeve correlation:  B↔C = {corr.loc['B', 'C']:+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    strategies = {"B": DonchianSleeveB(), "C": DonchianSleeveC()}

    print("=" * W)
    print("  IBKR FUTURES BACKTEST — Donchian Ensemble B+C")
    print(f"  Universe : {len(ALL_INSTRUMENTS)} instruments  "
          f"(Rates/Equity/FX/Metals/Energy/Agriculture)")
    print(f"  Sleeves  : B = Donchian(50)  |  C = Donchian(100)")
    print(f"  Risk     : {RISK_PER_TRADE*100:.1f}%/tr  1.5×ATR init  2×ATR trail  "
          f"{GLOBAL_CAP:.0f}R global cap")
    print("=" * W)

    results = run_all_periods(strategies, ALL_INSTRUMENTS)

    # ── Full period analysis ──────────────────────────────────────────────────
    full_label, full_start, full_end, full_eng, full_eq, full_trades, full_metrics = results[-1]

    print_portfolio_metrics(full_metrics, full_label)
    print_sleeve_stats(full_trades, full_start, full_end)
    print_asset_class(full_trades, full_label)
    print_regime_table(results)
    print_asset_class_by_period(results)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs("output/ibkr", exist_ok=True)
    full_eq.to_csv("output/ibkr/equity_curve.csv")
    full_trades.to_csv("output/ibkr/trades.csv", index=False)

    print(f"\n{'─'*W}")
    print(f"  Outputs: output/ibkr/  (equity_curve.csv, trades.csv)")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
