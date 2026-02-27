"""
stress/monte_carlo.py
─────────────────────
Monte Carlo reshuffling of realized trade PnL distribution.

Method:
  - Take actual closed-trade PnL values (dollar amounts).
  - Bootstrap-sample with replacement for N simulations.
  - Build equity path: eq[t+1] = eq[t] + sampled_pnl[t].
  - Compute CAGR using the full calendar duration of the original period.
  - Compute MaxDD for every simulated path.
  - Report tail percentiles across all simulations.

No rule changes, no resampling of entry timing or position sizing.
The distribution of outcomes reflects path-dependence in trade sequence only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


def run_monte_carlo(
    trades_df:      pd.DataFrame,
    initial_equity: float = 100_000.0,
    n_sims:         int   = 1000,
    seed:           int   = 42,
) -> Dict[str, Any]:
    """
    Bootstrap reshuffle of realized trade PnL sequence.

    Args:
        trades_df:      Closed trades DataFrame (must have 'pnl', 'entry_date', 'exit_date').
        initial_equity: Starting equity for path simulation.
        n_sims:         Number of Monte Carlo paths.
        seed:           Random seed for reproducibility.

    Returns:
        Dict with simulation statistics (CAGR percentiles, MaxDD percentiles,
        % profitable, % ruin). Returns {'error': ...} if insufficient data.
    """
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return {"error": "No trade data available"}

    pnl_vals = trades_df["pnl"].dropna().values
    n_trades  = len(pnl_vals)

    if n_trades < 10:
        return {"error": f"Insufficient trades for MC ({n_trades} trades, need ≥ 10)"}

    # ── Calendar duration from actual trade history ───────────────────────────
    if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
        entry_dates  = pd.to_datetime(trades_df["entry_date"])
        exit_dates   = pd.to_datetime(trades_df["exit_date"])
        period_start = entry_dates.min()
        period_end   = exit_dates.max()
        n_years      = max((period_end - period_start).days / 365.25, 0.5)
    else:
        n_years = 17.0  # fallback: full 2008–2025

    # ── Bootstrap simulation ──────────────────────────────────────────────────
    rng      = np.random.default_rng(seed)
    sim_pnl  = rng.choice(pnl_vals, size=(n_sims, n_trades), replace=True)

    # Equity paths: prepend initial equity, then cumsum
    eq_paths = np.hstack([
        np.full((n_sims, 1), initial_equity),
        initial_equity + np.cumsum(sim_pnl, axis=1),
    ])  # shape: (n_sims, n_trades + 1)

    # ── CAGR per simulation ───────────────────────────────────────────────────
    final_eq  = eq_paths[:, -1]
    ratio     = np.where(final_eq > 0, final_eq / initial_equity, 0.0)
    cagr_vals = np.where(ratio > 0, ratio ** (1.0 / n_years) - 1.0, -1.0)

    # ── MaxDD per simulation (vectorized) ────────────────────────────────────
    cum_max    = np.maximum.accumulate(eq_paths, axis=1)
    dd_matrix  = (eq_paths - cum_max) / cum_max
    max_dd_vals = dd_matrix.min(axis=1)   # most negative value per path

    # ── Ruin: equity fell below 50% of start ─────────────────────────────────
    pct_ruin = float((eq_paths.min(axis=1) < initial_equity * 0.50).mean())

    return {
        "n_simulations":    n_sims,
        "n_trades_per_sim": n_trades,
        "n_years":          round(n_years, 2),
        # CAGR distribution
        "cagr_p5":          float(np.percentile(cagr_vals, 5)),
        "cagr_p25":         float(np.percentile(cagr_vals, 25)),
        "cagr_p50":         float(np.percentile(cagr_vals, 50)),
        "cagr_p75":         float(np.percentile(cagr_vals, 75)),
        "cagr_p95":         float(np.percentile(cagr_vals, 95)),
        # MaxDD distribution
        "max_dd_p5":        float(np.percentile(max_dd_vals, 5)),
        "max_dd_p50":       float(np.percentile(max_dd_vals, 50)),
        "max_dd_p95":       float(np.percentile(max_dd_vals, 95)),
        # Summary
        "pct_profitable":   float((cagr_vals > 0).mean()),
        "pct_ruin":         pct_ruin,
    }
