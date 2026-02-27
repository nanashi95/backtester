"""
stress/runner.py
────────────────
Stress-test orchestration layer for Shizuka Core v1.

Wraps EnsemblePortfolioEngine with:
  - Date window overrides  (regime tests)
  - Parameter overrides    (sensitivity grid)
  - Universe filters       (market removal)
  - Portfolio cap overrides (heat stress)

Does NOT modify any engine, strategy, or risk rule.
Caller must configure data.loader before importing this module.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root is on path when this module is used standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.donchian_ensemble import _DonchianSleeve
from strategies.base import StrategyConfig
from engine.ensemble_engine import EnsemblePortfolioEngine
from metrics.metrics_engine import compute_all_metrics
from stress.metrics_helpers import extract_stress_metrics


# ── Production universe (mirrors zf_risk06_backtest.py) ──────────────────────

RATES       = ["ZF"]
EQUITY      = ["ES", "NQ", "RTY", "YM"]
METALS      = ["GC", "SI", "HG", "PA", "PL"]
ENERGY      = ["CL", "NG"]
AGRICULTURE = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

ALL_INSTRUMENTS: List[str] = RATES + EQUITY + METALS + ENERGY + AGRICULTURE

SECTOR_MAP: Dict[str, List[str]] = {
    "rates":       RATES,
    "equity":      EQUITY,
    "metals":      METALS,
    "energy":      ENERGY,
    "agriculture": AGRICULTURE,
}

# ── Production risk defaults ──────────────────────────────────────────────────

INITIAL_EQUITY     = 100_000.0
DEFAULT_RISK       = 0.006
DEFAULT_GLOBAL_CAP = 6.0
DEFAULT_ATR_INIT   = 1.5
DEFAULT_ATR_TRAIL  = 2.0

FULL_START = "2008-01-01"
FULL_END   = "2025-12-31"

# ── Regime test windows ───────────────────────────────────────────────────────

REGIME_WINDOWS: List[Tuple[str, str, str]] = [
    ("2008-01-01", "2009-12-31", "GFC 2008–2009"),
    ("2011-01-01", "2014-12-31", "Grinding range 2011–2014"),
    ("2015-01-01", "2016-12-31", "Whipsaw 2015–2016"),
    ("2018-01-01", "2018-12-31", "Risk-off 2018"),
    ("2020-01-01", "2020-12-31", "COVID 2020"),
    ("2022-01-01", "2022-12-31", "Rates shock 2022"),
]

# ── Parameter sensitivity grids ───────────────────────────────────────────────
# "Donchian 40/50/60" = base period for the B (medium) sleeves.
# All other sleeve periods scale proportionally from the D50 baseline.
# ATR multiple = trailing stop multiplier (initial stop stays at 1.5×).

DONCHIAN_BASE_PERIODS = [40, 50, 60]
ATR_TRAIL_MULTIPLES   = [1.8, 2.0, 2.2]
RISK_LEVELS           = [0.005, 0.007, 0.010]

# ── Heat stress caps ──────────────────────────────────────────────────────────

CAP_LEVELS = [4.0, 6.0, 8.0]


# ── Sleeve factory ────────────────────────────────────────────────────────────

def _make_sleeve(
    period:      int,
    label:       str,
    atr_initial: float = DEFAULT_ATR_INIT,
    atr_trail:   float = DEFAULT_ATR_TRAIL,
) -> _DonchianSleeve:
    """
    Return a _DonchianSleeve instance with overrideable ATR stop parameters.
    Uses a local subclass to override config() without touching the source file.
    """
    _period      = period
    _label       = label
    _atr_initial = atr_initial
    _atr_trail   = atr_trail

    class _OverriddenSleeve(_DonchianSleeve):
        def config(self) -> StrategyConfig:
            return StrategyConfig(
                name              = f"Sleeve {_label} — Donchian({_period})",
                atr_period        = 20,
                atr_initial_stop  = _atr_initial,
                atr_trailing_stop = _atr_trail,
                use_reversal_exit = False,
            )

    return _OverriddenSleeve(period=_period, label=_label)


def _make_strategies(
    base_period: int   = 50,
    atr_initial: float = DEFAULT_ATR_INIT,
    atr_trail:   float = DEFAULT_ATR_TRAIL,
) -> Dict[str, _DonchianSleeve]:
    """
    Build the 6-sleeve strategy dict.

    `base_period` scales all sleeve periods proportionally from the D50 origin:
      B sleeves  = base_period        (50  → 40 / 50 / 60)
      C sleeves  = base_period × 2    (100 → 80 / 100 / 120)
      B_EA       = base_period × 1.5  (75  → 60 / 75 / 90)
      C_EA       = base_period × 3    (150 → 120 / 150 / 180)
    """
    scale = base_period / 50.0
    return {
        "B_ME": _make_sleeve(round(50  * scale), "B", atr_initial, atr_trail),
        "C_ME": _make_sleeve(round(100 * scale), "C", atr_initial, atr_trail),
        "B_EA": _make_sleeve(round(75  * scale), "B", atr_initial, atr_trail),
        "C_EA": _make_sleeve(round(150 * scale), "C", atr_initial, atr_trail),
        "B_R":  _make_sleeve(round(50  * scale), "B", atr_initial, atr_trail),
        "C_R":  _make_sleeve(round(100 * scale), "C", atr_initial, atr_trail),
    }


def _build_sleeve_instruments(instruments: List[str]) -> Dict[str, List[str]]:
    """
    Rebuild sleeve→instrument mapping for a given instrument subset.
    Sleeves with zero instruments will simply not trade.
    """
    instr_set = set(instruments)
    return {
        "B_ME": [i for i in METALS + ENERGY      if i in instr_set],
        "C_ME": [i for i in METALS + ENERGY      if i in instr_set],
        "B_EA": [i for i in EQUITY + AGRICULTURE if i in instr_set],
        "C_EA": [i for i in EQUITY + AGRICULTURE if i in instr_set],
        "B_R":  [i for i in RATES                if i in instr_set],
        "C_R":  [i for i in RATES                if i in instr_set],
    }


# ── Core engine runner ────────────────────────────────────────────────────────

def _run_engine(
    start:        str,
    end:          str,
    instruments:  List[str]           = ALL_INSTRUMENTS,
    risk:         float               = DEFAULT_RISK,
    global_cap:   float               = DEFAULT_GLOBAL_CAP,
    base_period:  int                 = 50,
    atr_initial:  float               = DEFAULT_ATR_INIT,
    atr_trail:    float               = DEFAULT_ATR_TRAIL,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Single engine run. Returns (stress_metrics_dict, trades_df, equity_curve).
    All parameters default to the production baseline.
    """
    strategies    = _make_strategies(base_period, atr_initial, atr_trail)
    sleeve_instr  = _build_sleeve_instruments(instruments)

    eng = EnsemblePortfolioEngine(
        strategies         = strategies,
        start              = start,
        end                = end,
        initial_equity     = INITIAL_EQUITY,
        instruments        = instruments,
        risk_per_trade     = risk,
        global_cap         = global_cap,
        sleeve_instruments = sleeve_instr,
        sector_cap_pct     = 0.0,
        use_vol_scaling    = False,
    )

    eq     = eng.run()
    trades = eng.get_trades_df()
    m      = extract_stress_metrics(eq, trades, INITIAL_EQUITY)
    return m, trades, eq


# ── Regime stress tests ───────────────────────────────────────────────────────

def run_regime_tests() -> List[Dict[str, Any]]:
    """
    Run each regime window at production settings.
    Returns list of stress metric dicts, one per window.
    """
    results = []
    for start, end, label in REGIME_WINDOWS:
        print(f"    {label}...", flush=True)
        t0 = time.time()
        m, trades, eq = _run_engine(start, end)
        m["label"]   = label
        m["start"]   = start
        m["end"]     = end
        m["elapsed"] = round(time.time() - t0, 1)
        m["_trades"] = trades
        m["_equity"] = eq
        results.append(m)
    return results


# ── Parameter stability grid ──────────────────────────────────────────────────

def run_parameter_grid() -> Dict[str, List[Dict[str, Any]]]:
    """
    Run 3 independent single-dimension grids on the full 2008–2025 period.
    Returns dict keyed by dimension: 'donchian', 'atr_trail', 'risk'.
    """
    donchian_results: List[Dict[str, Any]] = []
    for period in DONCHIAN_BASE_PERIODS:
        print(f"    Donchian base D{period}...", flush=True)
        m, _, _ = _run_engine(FULL_START, FULL_END, base_period=period)
        m["param_label"] = f"D{period}"
        m["param_value"] = period
        donchian_results.append(m)

    atr_results: List[Dict[str, Any]] = []
    for atr in ATR_TRAIL_MULTIPLES:
        print(f"    ATR trail {atr}×...", flush=True)
        m, _, _ = _run_engine(FULL_START, FULL_END, atr_trail=atr)
        m["param_label"] = f"{atr}×ATR"
        m["param_value"] = atr
        atr_results.append(m)

    risk_results: List[Dict[str, Any]] = []
    for risk in RISK_LEVELS:
        print(f"    Risk {risk:.1%}...", flush=True)
        m, _, _ = _run_engine(FULL_START, FULL_END, risk=risk)
        m["param_label"] = f"{risk:.1%}"
        m["param_value"] = risk
        risk_results.append(m)

    return {
        "donchian":  donchian_results,
        "atr_trail": atr_results,
        "risk":      risk_results,
    }


# ── Market removal tests ──────────────────────────────────────────────────────

def run_market_removal(baseline_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Remove each asset class one at a time. Measures CAGR and MaxDD delta.
    Returns list of result dicts, one per removed sector.
    """
    results = []
    for sector, removed in SECTOR_MAP.items():
        remaining = [i for i in ALL_INSTRUMENTS if i not in removed]
        print(f"    Remove {sector} ({len(removed)} instruments, {len(remaining)} remain)...", flush=True)
        m, _, _ = _run_engine(FULL_START, FULL_END, instruments=remaining)
        m["removed_sector"] = sector
        m["removed_count"]  = len(removed)
        m["cagr_delta"]     = m["cagr"]   - baseline_metrics["cagr"]
        m["max_dd_delta"]   = m["max_dd"] - baseline_metrics["max_dd"]
        results.append(m)
    return results


# ── Heat stress tests ─────────────────────────────────────────────────────────

def run_heat_stress() -> List[Dict[str, Any]]:
    """
    Test portfolio cap at 4R / 6R / 8R on the full period.
    Returns list of result dicts, one per cap level.
    """
    results = []
    for cap in CAP_LEVELS:
        print(f"    Global cap {cap:.0f}R...", flush=True)
        m, _, _ = _run_engine(FULL_START, FULL_END, global_cap=cap)
        m["cap_level"] = cap
        results.append(m)
    return results


# ── Full orchestration ────────────────────────────────────────────────────────

def run_all() -> Dict[str, Any]:
    """
    Execute the complete stress test suite.

    Run order:
      1. Baseline (2008–2025, production settings)
      2. Regime tests (6 windows)
      3. Parameter stability (3 grids × 3 variants = 9 runs)
      4. Market removal (5 sectors)
      5. Heat stress (3 cap levels)

    Total engine runs: ~24. Monte Carlo is handled separately by the caller.

    Returns structured dict:
      {
        'baseline':   {'metrics': ..., 'trades': ..., 'equity': ...},
        'regime':     [...],
        'parameters': {'donchian': [...], 'atr_trail': [...], 'risk': [...]},
        'removal':    [...],
        'heat':       [...],
      }
    """
    W = 72
    print("\n" + "═" * W)
    print("  SHIZUKA CORE v1 — STRUCTURAL STRESS TEST SUITE")
    print("  Objective: Diagnostics only. No optimization. No rule changes.")
    print("═" * W)

    print("\n[1/5] Baseline 2008–2025 (production settings)...", flush=True)
    t0 = time.time()
    baseline_m, baseline_trades, baseline_eq = _run_engine(FULL_START, FULL_END)
    baseline_m["label"] = "Baseline 2008–2025"
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[2/5] Regime stress tests (6 windows)...", flush=True)
    regime_results = run_regime_tests()

    print("\n[3/5] Parameter stability grid (9 runs)...", flush=True)
    param_results = run_parameter_grid()

    print("\n[4/5] Market removal tests (5 sectors)...", flush=True)
    removal_results = run_market_removal(baseline_m)

    print("\n[5/5] Heat stress — portfolio cap (3 levels)...", flush=True)
    heat_results = run_heat_stress()

    return {
        "baseline": {
            "metrics": baseline_m,
            "trades":  baseline_trades,
            "equity":  baseline_eq,
        },
        "regime":     regime_results,
        "parameters": param_results,
        "removal":    removal_results,
        "heat":       heat_results,
    }
