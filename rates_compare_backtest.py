"""
rates_compare_backtest.py
-------------------------
Two rates configurations tested against the D150/D300 baseline:

  A) ZN + ZF only  (drop ZB),  B=D50  C=D100
  B) ZN + ZB + ZF  (all three), B=D75  C=D150

All other settings identical: Metals/Energy D50/100, Equity/Agri D75/150,
no FX, no vol-scaling, no sector caps.
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

EQUITY      = ["ES", "NQ", "RTY", "YM"]
METALS      = ["GC", "SI", "HG", "PA", "PL"]
ENERGY      = ["CL", "NG"]
AGRICULTURE = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

METALS_ENERGY = METALS + ENERGY
EQUITY_AGRI   = EQUITY + AGRICULTURE

BUCKET_ORDER  = ["rates", "equity", "metals", "energy", "agriculture"]
BUCKET_LABELS = {"rates": "Rates", "equity": "Equity", "metals": "Metals",
                 "energy": "Energy", "agriculture": "Agriculture"}

INITIAL_EQUITY = 100_000.0
RISK_PER_TRADE = 0.007
GLOBAL_CAP     = 6.0
START, END     = "2008-01-01", "2025-12-31"
W = 72

PERIODS = [
    ("2008-01-01", "2016-12-31", "2008–2016"),
    ("2017-01-01", "2022-12-31", "2017–2022"),
    ("2023-01-01", "2025-12-31", "2023–2025"),
    ("2008-01-01", "2025-12-31", "2008–2025"),
]

BASELINE = {
    "label":                   "D150/D300 (ZN+ZB+ZF)",
    "cagr":                     0.0969,
    "max_drawdown":            -0.3098,
    "mar_ratio":                0.313,
    "sharpe":                   0.50,
    "longest_underwater_days":  2010,
    "rates_pnl":               -21724.0,
}

# ── Configs ───────────────────────────────────────────────────────────────────

CONFIGS = {
    "A_ZN_ZF_D50":  {
        "label":        "A: ZN+ZF  D50/D100",
        "rates":        ["ZN", "ZF"],
        "rates_b":      50,
        "rates_c":      100,
    },
    "B_all_D75": {
        "label":        "B: ZN+ZB+ZF  D75/D150",
        "rates":        ["ZN", "ZB", "ZF"],
        "rates_b":      75,
        "rates_c":      150,
    },
}

# ── Runner ────────────────────────────────────────────────────────────────────

def make_strategies(rates_b, rates_c):
    return {
        "B_ME": make_sleeve(50,       "B"),
        "C_ME": make_sleeve(100,      "C"),
        "B_EA": make_sleeve(75,       "B"),
        "C_EA": make_sleeve(150,      "C"),
        "B_R":  make_sleeve(rates_b,  "B"),
        "C_R":  make_sleeve(rates_c,  "C"),
    }

def run(cfg: dict, start: str, end: str):
    rates = cfg["rates"]
    all_instruments = rates + EQUITY + METALS + ENERGY + AGRICULTURE
    si = {
        "B_ME": METALS_ENERGY, "C_ME": METALS_ENERGY,
        "B_EA": EQUITY_AGRI,   "C_EA": EQUITY_AGRI,
        "B_R":  rates,         "C_R":  rates,
    }
    eng = EnsemblePortfolioEngine(
        strategies         = make_strategies(cfg["rates_b"], cfg["rates_c"]),
        start              = start,
        end                = end,
        initial_equity     = INITIAL_EQUITY,
        instruments        = all_instruments,
        risk_per_trade     = RISK_PER_TRADE,
        global_cap         = GLOBAL_CAP,
        sleeve_instruments = si,
        use_vol_scaling    = False,
    )
    eq     = eng.run()
    trades = eng.get_trades_df()
    return eq, trades

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_f(v, d=2):
    return f"{v:.{d}f}" if v == v else " nan"

def ann_vol(eq):
    s = eq["total_value"].resample("D").last().ffill()
    dr = s.pct_change().dropna()
    return float(dr.std() * np.sqrt(252) * 100) if len(dr) > 1 else float("nan")

def rates_pnl(trades):
    return trades[trades["bucket"] == "rates"]["pnl"].sum()

def rates_detail(trades, rates_instruments):
    rows = []
    for instr in rates_instruments:
        seg = trades[trades["instrument"] == instr]
        if seg.empty:
            rows.append(f"    {instr:<6s}  — no trades —")
            continue
        r   = seg["r_multiple"].dropna().values
        pnl = seg["pnl"].sum()
        exp = float(np.mean(r)) if len(r) else float("nan")
        wr  = 100.0 * float(np.mean(r > 0)) if len(r) else float("nan")
        rows.append(f"    {instr:<6s}  {len(seg):>5d} trades  "
                    f"{exp:>+6.3f}R  {wr:>5.1f}%  {pnl:>+10.0f}$")
    return "\n".join(rows)

# ── Print ──────────────────────────────────────────────────────────────────────

def print_side_by_side(results: dict):
    """Full-period metrics table: baseline + config A + config B."""
    keys  = list(results.keys())
    labels = [results[k]["label"] for k in keys]
    metrics_list = [results[k]["metrics"] for k in keys]
    eq_list      = [results[k]["eq"]      for k in keys]
    trades_list  = [results[k]["trades"]  for k in keys]

    col_w = 20

    print(f"\n{'═'*W}")
    print(f"  FULL-PERIOD COMPARISON — 2008–2025")
    print(f"{'═'*W}")
    print(f"  {'Metric':<22s}", end="")
    print(f"  {'Baseline':>{col_w}s}", end="")
    for lbl in labels:
        print(f"  {lbl:>{col_w}s}", end="")
    print()
    print(f"  {'─'*(22 + (col_w+2) * (len(keys)+1))}")

    def row(label, fn, fmt=".2f", suffix=""):
        print(f"  {label:<22s}", end="")
        bv = fn(None, None, BASELINE)
        print(f"  {bv:>{col_w}s}", end="")
        for m, eq, tr in zip(metrics_list, eq_list, trades_list):
            v = fn(m, eq, tr)
            print(f"  {v:>{col_w}s}", end="")
        print()

    def _cagr(m, eq, b):
        if b: return f"{b['cagr']*100:+.2f}%"
        return f"{m['cagr']*100:+.2f}%"
    def _dd(m, eq, b):
        if b: return f"{b['max_drawdown']*100:.2f}%"
        return f"{m['max_drawdown']*100:.2f}%"
    def _mar(m, eq, b):
        if b: return fmt_f(b['mar_ratio'], 3)
        return fmt_f(m.get('mar_ratio', float('nan')), 3)
    def _sharpe(m, eq, b):
        if b: return fmt_f(b['sharpe'], 2)
        return fmt_f(m.get('sharpe', float('nan')), 2)
    def _uw(m, eq, b):
        if b: return f"{b['longest_underwater_days']}d"
        return f"{m.get('longest_underwater_days', 0)}d"
    def _vol(m, eq, b):
        if b: return "—"
        return f"{fmt_f(ann_vol(eq), 2)}%"
    def _trades(m, eq, b):
        if b: return "—"
        n_yr = m.get('n_years', 1)
        return f"{m.get('total_trades',0)/n_yr:.0f}/yr"
    def _rates_pnl(m, eq, b):
        if b: return f"${b['rates_pnl']:+,.0f}"
        return f"${rates_pnl(eq):.0f}"   # eq is trades here via closure trick
    def _wr(m, eq, b):
        if b: return "—"
        return f"{m.get('win_rate',0)*100:.1f}%"
    def _pf(m, eq, b):
        if b: return "—"
        return fmt_f(m.get('profit_factor', float('nan')), 2)

    # Use separate call pattern
    headers = [
        ("CAGR",        lambda m,eq,tr,b: f"{b['cagr']*100:+.2f}%" if b else f"{m['cagr']*100:+.2f}%"),
        ("MaxDD",       lambda m,eq,tr,b: f"{b['max_drawdown']*100:.2f}%" if b else f"{m['max_drawdown']*100:.2f}%"),
        ("MAR",         lambda m,eq,tr,b: fmt_f(b['mar_ratio'],3) if b else fmt_f(m.get('mar_ratio',float('nan')),3)),
        ("Sharpe",      lambda m,eq,tr,b: fmt_f(b['sharpe'],2) if b else fmt_f(m.get('sharpe',float('nan')),2)),
        ("Longest UW",  lambda m,eq,tr,b: f"{b['longest_underwater_days']}d" if b else f"{m.get('longest_underwater_days',0)}d"),
        ("Ann. Vol",    lambda m,eq,tr,b: "—" if b else f"{fmt_f(ann_vol(eq),2)}%"),
        ("Trades/yr",   lambda m,eq,tr,b: "—" if b else f"{m.get('total_trades',0)/m.get('n_years',1):.0f}"),
        ("Win Rate",    lambda m,eq,tr,b: "—" if b else f"{m.get('win_rate',0)*100:.1f}%"),
        ("Profit Factor", lambda m,eq,tr,b: "—" if b else fmt_f(m.get('profit_factor',float('nan')),2)),
        ("Rates PnL",   lambda m,eq,tr,b: f"${b['rates_pnl']:+,.0f}" if b else f"${rates_pnl(tr):+,.0f}"),
    ]

    for lbl, fn in headers:
        print(f"  {lbl:<22s}", end="")
        print(f"  {fn(None,None,None,BASELINE):>{col_w}s}", end="")
        for m, eq, tr in zip(metrics_list, eq_list, trades_list):
            print(f"  {fn(m,eq,tr,None):>{col_w}s}", end="")
        print()


def print_regime_table(regime_results: dict):
    """Per-period breakdown for each config."""
    print(f"\n{'─'*W}")
    print(f"  REGIME TABLE")
    print(f"{'─'*W}")

    col_w = 14
    configs = list(regime_results.keys())

    # Header
    print(f"  {'Period':<12s}", end="")
    for cfg_key in configs:
        lbl = regime_results[cfg_key]["label"]
        print(f"  {lbl[:col_w]:>{col_w}s}", end="")
    print()
    sub = "  CAGR / MaxDD / Sharpe"
    print(f"  {'─'*W}")

    for _, _, period_label in PERIODS:
        print(f"  {period_label:<12s}", end="")
        for cfg_key in configs:
            pr = regime_results[cfg_key]["periods"].get(period_label, {})
            cagr = pr.get("cagr", float("nan"))
            dd   = pr.get("max_drawdown", float("nan"))
            sh   = pr.get("sharpe", float("nan"))
            if cagr == cagr:
                cell = f"{cagr*100:+.1f}% / {dd*100:.1f}% / {fmt_f(sh,2)}"
            else:
                cell = "—"
            print(f"  {cell:>{col_w+2}s}", end="")
        print()


def print_rates_detail(results: dict):
    print(f"\n{'─'*W}")
    print(f"  RATES INSTRUMENTS DETAIL — 2008–2025")
    print(f"{'─'*W}")
    for cfg_key, res in results.items():
        cfg = CONFIGS[cfg_key]
        print(f"\n  [{res['label']}]")
        print(rates_detail(res["trades"], cfg["rates"]))


def print_asset_class_row(results: dict):
    """Asset class PnL for each config side by side."""
    print(f"\n{'─'*W}")
    print(f"  ASSET CLASS PnL — 2008–2025")
    print(f"{'─'*W}")

    configs = list(results.keys())
    labels  = [results[k]["label"] for k in configs]
    col_w   = 18

    print(f"  {'Sector':<14s}", end="")
    print(f"  {'Baseline':>{col_w}s}", end="")
    for lbl in labels:
        print(f"  {lbl[:col_w]:>{col_w}s}", end="")
    print()
    print(f"  {'─'*(14 + (col_w+2) * (len(configs)+1))}")

    BASE_SECTOR_PNL = {
        "rates": -21724, "equity": -18305, "metals": 409225,
        "energy": -28586, "agriculture": -29811,
    }  # from speed-diff baseline

    for bkt in BUCKET_ORDER:
        lbl = BUCKET_LABELS[bkt]
        base_pnl = BASE_SECTOR_PNL.get(bkt, 0)
        print(f"  {lbl:<14s}", end="")
        print(f"  {f'${base_pnl:+,.0f}':>{col_w}s}", end="")
        for cfg_key in configs:
            tr = results[cfg_key]["trades"]
            pnl = tr[tr["bucket"] == bkt]["pnl"].sum()
            print(f"  {f'${pnl:+,.0f}':>{col_w}s}", end="")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * W)
    print("  RATES CONFIG COMPARISON")
    print(f"  A: ZN+ZF only,      Rates B=D50  C=D100")
    print(f"  B: ZN+ZB+ZF (all),  Rates B=D75  C=D150")
    print(f"  Baseline: D150/D300 (all three) — hardcoded from prior run")
    print("=" * W)

    full_results   = {}
    regime_results = {}

    for cfg_key, cfg in CONFIGS.items():
        print(f"\n{'─'*W}")
        print(f"  Config {cfg['label']} — full period")
        print(f"{'─'*W}")
        eq, trades = run(cfg, START, END)
        metrics    = compute_all_metrics(eq, trades, INITIAL_EQUITY)
        full_results[cfg_key] = {
            "label":   cfg["label"],
            "eq":      eq,
            "trades":  trades,
            "metrics": metrics,
        }

        # Per-period
        period_metrics = {}
        for start, end, label in PERIODS:
            print(f"  Running {label}...")
            eq_p, tr_p = run(cfg, start, end)
            m_p = compute_all_metrics(eq_p, tr_p, INITIAL_EQUITY)
            period_metrics[label] = m_p

        regime_results[cfg_key] = {
            "label":   cfg["label"],
            "periods": period_metrics,
        }

    # ── Output ────────────────────────────────────────────────────────────────
    print_side_by_side(full_results)
    print_regime_table(regime_results)
    print_rates_detail(full_results)
    print_asset_class_row(full_results)

    # Save
    os.makedirs("output/rates_compare", exist_ok=True)
    for cfg_key, res in full_results.items():
        res["eq"].to_csv(f"output/rates_compare/equity_{cfg_key}.csv")
        res["trades"].to_csv(f"output/rates_compare/trades_{cfg_key}.csv", index=False)

    print(f"\n{'─'*W}")
    print(f"  Outputs: output/rates_compare/")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
