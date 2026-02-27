"""
stress/metrics_helpers.py
─────────────────────────
Extended metric helpers for stress testing.

Adds metrics not in the base metrics_engine:
  - worst_losing_streak_r   : cumulative R of worst consecutive losing run
  - max_consecutive_losses  : count of worst losing streak
  - pct_time_in_drawdown    : fraction of days below all-time high
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Dict, Any

from metrics.metrics_engine import compute_all_metrics


def worst_losing_streak_r(trades_df: pd.DataFrame) -> float:
    """
    Maximum cumulative R loss across any consecutive losing streak.
    Returns a negative number (worst = most negative). Returns 0.0 if no trades.
    """
    if trades_df.empty or "r_multiple" not in trades_df.columns:
        return 0.0
    rs = trades_df["r_multiple"].dropna().values
    worst   = 0.0
    running = 0.0
    for r in rs:
        if r < 0:
            running += r
            worst = min(worst, running)
        else:
            running = 0.0
    return float(worst)


def max_consecutive_losses(trades_df: pd.DataFrame) -> int:
    """Maximum number of consecutive losing trades."""
    if trades_df.empty or "r_multiple" not in trades_df.columns:
        return 0
    rs    = trades_df["r_multiple"].dropna().values
    best  = 0
    count = 0
    for r in rs:
        if r < 0:
            count += 1
            best = max(best, count)
        else:
            count = 0
    return int(best)


def pct_time_in_drawdown(equity_curve: pd.DataFrame) -> float:
    """
    Fraction of trading days where equity < previous all-time high.
    Uses the total_value column (settled cash + unrealised MTM).
    """
    if equity_curve.empty or "total_value" not in equity_curve.columns:
        return 0.0
    eq          = equity_curve["total_value"]
    rolling_max = eq.cummax()
    in_dd       = eq < rolling_max * 0.9999
    return float(in_dd.mean())


def extract_stress_metrics(
    equity_curve:   pd.DataFrame,
    trades_df:      pd.DataFrame,
    initial_equity: float = 100_000.0,
) -> Dict[str, Any]:
    """
    Compute base metrics (via metrics_engine) + stress-specific extensions.
    Returns a flat dict with all metrics needed by the stress report.
    """
    base = compute_all_metrics(equity_curve, trades_df, initial_equity)

    m: Dict[str, Any] = {}

    # ── Core (flattened from base) ────────────────────────────────────────────
    m["cagr"]          = base["cagr"]
    m["max_dd"]        = base["max_drawdown"]
    m["mar"]           = base.get("mar_ratio", float("nan"))
    m["sharpe"]        = base.get("sharpe", float("nan"))
    m["profit_factor"] = base.get("profit_factor", float("nan"))
    m["total_trades"]  = base.get("total_trades", 0)
    m["win_rate"]      = base.get("win_rate", 0.0)
    m["avg_win_r"]     = base.get("avg_win_r", 0.0)
    m["avg_loss_r"]    = base.get("avg_loss_r", 0.0)
    m["expectancy_r"]  = base.get("expectancy_r", 0.0)
    m["avg_hold_days"] = base.get("avg_hold_days", 0.0)
    m["longest_uw_days"]  = base.get("longest_underwater_days", 0)
    m["annual_returns"]   = base.get("annual_returns", {})
    m["bucket_breakdown"] = base.get("bucket_breakdown", {})
    m["trades_per_year"]  = base.get("trades_per_year", {})
    m["n_years"]          = base.get("n_years", 0.0)

    n_years = m["n_years"] or 1.0
    m["trades_per_year_avg"] = m["total_trades"] / n_years if n_years > 0 else 0.0

    # ── Stress extensions ─────────────────────────────────────────────────────
    m["worst_streak_r"]  = worst_losing_streak_r(trades_df)
    m["max_consec_loss"] = max_consecutive_losses(trades_df)
    m["pct_time_in_dd"]  = pct_time_in_drawdown(equity_curve)

    return m
