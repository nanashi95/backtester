"""
metrics_engine.py
-----------------
Computes all required backtest performance metrics.

Outputs:
  - CAGR
  - Max equity drawdown (peak-to-trough)
  - MAR ratio (CAGR / Max DD)
  - Longest time under water (days)
  - Worst rolling 12-month return
  - Annual returns breakdown
  - Trade count per year
  - Win rate
  - Avg win / avg loss
  - R distribution
  - Bucket performance breakdown
  - Exposure clustering analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def compute_all_metrics(equity_curve: pd.DataFrame,
                        trades_df: pd.DataFrame,
                        initial_equity: float = 100_000.0) -> Dict[str, Any]:
    """Master function: compute and return all metrics as a dict."""
    metrics: Dict[str, Any] = {}

    # ── Equity series (use total_value = cash + unrealised MTM) ─────────────
    eq = equity_curve["total_value"].copy()
    eq.index = pd.to_datetime(eq.index)

    # ── CAGR ─────────────────────────────────────────────────────────────────
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    if n_years > 0 and eq.iloc[0] > 0:
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1
    else:
        cagr = 0.0
    metrics["cagr"]   = cagr
    metrics["n_years"] = n_years

    # ── Max drawdown (peak-to-trough) ─────────────────────────────────────────
    rolling_max = eq.cummax()
    drawdowns   = (eq - rolling_max) / rolling_max
    max_dd      = drawdowns.min()
    metrics["max_drawdown"] = max_dd

    # ── MAR ratio ────────────────────────────────────────────────────────────
    metrics["mar_ratio"] = (cagr / abs(max_dd)) if max_dd != 0 else np.nan

    # ── Sharpe ratio (annualised, risk-free rate = 0) ─────────────────────────
    eq_daily    = eq.resample("D").last().ffill()
    daily_rets  = eq_daily.pct_change().dropna()
    if len(daily_rets) > 1 and daily_rets.std() > 0:
        sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
    else:
        sharpe = np.nan
    metrics["sharpe"] = sharpe

    # ── Longest time underwater ───────────────────────────────────────────────
    # "Underwater" = equity below previous all-time high
    at_high = eq >= rolling_max * 0.9999  # tolerance for float equality
    uw_streaks = []
    in_uw, streak_start = False, None
    for date, is_at_high in at_high.items():
        if not is_at_high and not in_uw:
            in_uw = True
            streak_start = date
        elif is_at_high and in_uw:
            uw_streaks.append((streak_start, date, (date - streak_start).days))
            in_uw = False
    if in_uw:
        uw_streaks.append((streak_start, eq.index[-1],
                           (eq.index[-1] - streak_start).days))
    metrics["longest_underwater_days"] = max((s[2] for s in uw_streaks), default=0)
    metrics["underwater_periods"]      = uw_streaks

    # ── Worst rolling 12-month return ────────────────────────────────────────
    eq_daily = eq.resample("D").last().ffill()
    roll_12m  = eq_daily.pct_change(periods=252)
    metrics["worst_12m_return"] = float(roll_12m.min())

    # ── Annual returns breakdown ──────────────────────────────────────────────
    annual_eq   = eq.resample("YE").last()
    annual_eq   = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), annual_eq])
    annual_rets = annual_eq.pct_change().dropna()
    annual_rets.index = annual_rets.index.year
    metrics["annual_returns"] = annual_rets.to_dict()

    # ── Trade count per year ──────────────────────────────────────────────────
    if not trades_df.empty and "entry_date" in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
        metrics["trades_per_year"] = trades_df.groupby("year").size().to_dict()
    else:
        metrics["trades_per_year"] = {}

    # ── Win rate / avg win / avg loss ─────────────────────────────────────────
    if not trades_df.empty and "r_multiple" in trades_df.columns:
        closed = trades_df[trades_df["r_multiple"].notna()].copy()
        n_trades  = len(closed)
        wins      = closed[closed["r_multiple"] > 0]
        losses    = closed[closed["r_multiple"] <= 0]

        metrics["total_trades"] = n_trades
        metrics["win_rate"]     = len(wins) / n_trades if n_trades > 0 else 0.0
        metrics["avg_win_r"]    = float(wins["r_multiple"].mean()) if len(wins) > 0 else 0.0
        metrics["avg_loss_r"]   = float(losses["r_multiple"].mean()) if len(losses) > 0 else 0.0
        metrics["profit_factor"] = (
            (wins["r_multiple"].sum() / abs(losses["r_multiple"].sum()))
            if losses["r_multiple"].sum() != 0 else np.inf
        )
        metrics["expectancy_r"] = (
            metrics["win_rate"] * metrics["avg_win_r"] +
            (1 - metrics["win_rate"]) * metrics["avg_loss_r"]
        )
        metrics["avg_hold_days"] = float(closed["hold_days"].mean()) if "hold_days" in closed else 0.0
    else:
        metrics.update({
            "total_trades": 0, "win_rate": 0, "avg_win_r": 0,
            "avg_loss_r": 0, "profit_factor": 0, "expectancy_r": 0,
        })

    # ── R distribution ───────────────────────────────────────────────────────
    if not trades_df.empty:
        r_vals = trades_df["r_multiple"].dropna()
        bins   = [-np.inf, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 5, np.inf]
        labels = ["<-3R", "-3→-2R", "-2→-1.5R", "-1.5→-1R", "-1→-0.5R", "-0.5→0R",
                  "0→0.5R", "0.5→1R", "1→1.5R", "1.5→2R", "2→3R", "3→5R", ">5R"]
        r_dist = pd.cut(r_vals, bins=bins, labels=labels).value_counts().sort_index()
        metrics["r_distribution"] = r_dist.to_dict()
        metrics["r_percentiles"]  = {
            "p5":    float(np.percentile(r_vals, 5)),
            "p25":   float(np.percentile(r_vals, 25)),
            "p50":   float(np.percentile(r_vals, 50)),
            "p75":   float(np.percentile(r_vals, 75)),
            "p95":   float(np.percentile(r_vals, 95)),
            "max":   float(r_vals.max()),
            "min":   float(r_vals.min()),
            "stdev": float(r_vals.std()),
        }

    # ── Bucket performance breakdown ─────────────────────────────────────────
    if not trades_df.empty and "bucket" in trades_df.columns:
        bucket_metrics = {}
        for bucket, grp in trades_df.groupby("bucket"):
            bm: Dict[str, Any] = {}
            bm["trade_count"] = len(grp)
            r  = grp["r_multiple"].dropna()
            bm["win_rate"]     = float((r > 0).mean()) if len(r) > 0 else 0
            bm["avg_r"]        = float(r.mean())       if len(r) > 0 else 0
            bm["total_r"]      = float(r.sum())        if len(r) > 0 else 0
            bm["avg_hold"]     = float(grp["hold_days"].mean()) if "hold_days" in grp else 0
            bucket_metrics[bucket] = bm
        metrics["bucket_breakdown"] = bucket_metrics

    # ── Exposure clustering analysis ─────────────────────────────────────────
    metrics["exposure_clustering"] = _exposure_clustering(equity_curve, trades_df)

    return metrics


def _exposure_clustering(equity_curve: pd.DataFrame,
                         trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse how often we're near or at the portfolio cap."""
    if equity_curve.empty:
        return {}
    r_open = equity_curve["total_r_open"]
    return {
        "avg_r_open":         float(r_open.mean()),
        "max_r_open":         float(r_open.max()),
        "pct_days_above_4r":  float((r_open >= 4).mean()),
        "pct_days_above_5r":  float((r_open >= 5).mean()),
        "pct_days_at_6r_cap": float((r_open >= 5.9).mean()),
        "avg_open_trades":    float(equity_curve["open_trades"].mean()),
        "max_open_trades":    int(equity_curve["open_trades"].max()),
    }


def format_report(metrics: Dict[str, Any]) -> str:
    """Format metrics into a clean console report."""
    lines = []
    sep   = "═" * 60

    lines.append(sep)
    lines.append("  TREND FOLLOWING BACKTEST  ─  PERFORMANCE REPORT")
    lines.append(sep)

    lines.append("\n─── PORTFOLIO SUMMARY ──────────────────────────────────")
    lines.append(f"  CAGR                    : {metrics['cagr']*100:>8.2f}%")
    lines.append(f"  Max Drawdown            : {metrics['max_drawdown']*100:>8.2f}%")
    lines.append(f"  MAR Ratio               : {metrics.get('mar_ratio', 0):>8.2f}x")
    sharpe = metrics.get("sharpe", float("nan"))
    sharpe_str = f"{sharpe:>8.2f}" if not np.isnan(sharpe) else "     nan"
    lines.append(f"  Sharpe Ratio            : {sharpe_str}")
    lines.append(f"  Longest Underwater      : {metrics['longest_underwater_days']:>8d} days")
    lines.append(f"  Worst 12-month Return   : {metrics['worst_12m_return']*100:>8.2f}%")

    lines.append("\n─── TRADE STATISTICS ───────────────────────────────────")
    lines.append(f"  Total Trades            : {metrics['total_trades']:>8d}")
    lines.append(f"  Win Rate                : {metrics['win_rate']*100:>8.2f}%")
    lines.append(f"  Avg Win (R)             : {metrics['avg_win_r']:>8.2f}R")
    lines.append(f"  Avg Loss (R)            : {metrics['avg_loss_r']:>8.2f}R")
    lines.append(f"  Profit Factor           : {metrics['profit_factor']:>8.2f}")
    lines.append(f"  Expectancy              : {metrics['expectancy_r']:>8.2f}R")
    lines.append(f"  Avg Hold (days)         : {metrics.get('avg_hold_days', 0):>8.1f}")

    lines.append("\n─── ANNUAL RETURNS ─────────────────────────────────────")
    for year, ret in sorted(metrics["annual_returns"].items()):
        bar = "█" * int(abs(ret) * 100 / 2)
        sign = "+" if ret >= 0 else ""
        lines.append(f"  {year}  {sign}{ret*100:>6.2f}%  {bar}")

    lines.append("\n─── TRADES PER YEAR ────────────────────────────────────")
    for year, cnt in sorted(metrics["trades_per_year"].items()):
        lines.append(f"  {year}  {cnt:>4d} trades")

    lines.append("\n─── R DISTRIBUTION ─────────────────────────────────────")
    r_dist = metrics.get("r_distribution", {})
    total  = sum(r_dist.values()) or 1
    for label, cnt in r_dist.items():
        pct = cnt / total * 100
        bar = "▪" * max(1, int(pct / 2))
        lines.append(f"  {label:<12s} {cnt:>5d} ({pct:>5.1f}%)  {bar}")

    pcts = metrics.get("r_percentiles", {})
    if pcts:
        lines.append(f"\n  R Percentiles  P5:{pcts['p5']:.2f}  P25:{pcts['p25']:.2f}"
                     f"  Median:{pcts['p50']:.2f}  P75:{pcts['p75']:.2f}  P95:{pcts['p95']:.2f}")
        lines.append(f"  R Std Dev: {pcts['stdev']:.2f}   Min: {pcts['min']:.2f}   Max: {pcts['max']:.2f}")

    lines.append("\n─── BUCKET BREAKDOWN ───────────────────────────────────")
    for bucket, bm in metrics.get("bucket_breakdown", {}).items():
        lines.append(f"  {bucket:<12s}  Trades:{bm['trade_count']:>4d}"
                     f"  WR:{bm['win_rate']*100:>5.1f}%"
                     f"  Avg R:{bm['avg_r']:>5.2f}"
                     f"  Total R:{bm['total_r']:>7.1f}")

    lines.append("\n─── EXPOSURE CLUSTERING ────────────────────────────────")
    ec = metrics.get("exposure_clustering", {})
    lines.append(f"  Avg R Open              : {ec.get('avg_r_open', 0):>8.2f}R")
    lines.append(f"  Max R Open              : {ec.get('max_r_open', 0):>8.2f}R")
    lines.append(f"  Days ≥ 4R Open          : {ec.get('pct_days_above_4r', 0)*100:>7.1f}%")
    lines.append(f"  Days ≥ 5R Open          : {ec.get('pct_days_above_5r', 0)*100:>7.1f}%")
    lines.append(f"  Days at 6R Cap          : {ec.get('pct_days_at_6r_cap', 0)*100:>7.1f}%")
    lines.append(f"  Avg Open Trades         : {ec.get('avg_open_trades', 0):>8.1f}")
    lines.append(f"  Max Concurrent Trades   : {ec.get('max_open_trades', 0):>8d}")

    lines.append("\n" + sep)
    return "\n".join(lines)
