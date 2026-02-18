"""
trade_audit.py
--------------
Debug tool: replays trailing stop bar-by-bar for 3 auto-selected trades.

Selections:
  1. Big winner     -- highest r_multiple
  2. Typical loser  -- r_multiple closest to median of all losers
  3. Long flat      -- longest hold_days where abs(r_multiple) < 0.3

Usage:
    python trade_audit.py                          # uses first strategy
    python trade_audit.py "SMA(50/200) Crossover"  # pick by name
"""

import sys
import os
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.mt5_data_loader import load_all_data
from engine.signal_engine import precompute_indicators
from strategies import STRATEGIES
from strategies.base import Strategy


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _find_strategy(name: str | None) -> Strategy:
    """Find strategy by name, or return first if name is None."""
    if name is None:
        return STRATEGIES[0]
    for s in STRATEGIES:
        if s.config().name == name:
            return s
    raise ValueError(f"Strategy '{name}' not found. "
                     f"Available: {[s.config().name for s in STRATEGIES]}")


# ── Load trades & indicator data ─────────────────────────────────────────────

def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    if "hold_days" not in df.columns:
        df["hold_days"] = (df["exit_date"] - df["entry_date"]).dt.days
    return df


def select_audit_trades(trades: pd.DataFrame) -> list[dict]:
    """Pick 3 representative trades."""
    selections = []

    # 1. Big winner -- highest r_multiple
    idx_winner = trades["r_multiple"].idxmax()
    selections.append({"label": "BIG WINNER", "row": trades.loc[idx_winner]})

    # 2. Typical loser -- closest to median of losing trades
    losers = trades[trades["r_multiple"] < 0]
    if not losers.empty:
        median_r = losers["r_multiple"].median()
        idx_loser = (losers["r_multiple"] - median_r).abs().idxmin()
        selections.append({"label": "TYPICAL LOSER", "row": trades.loc[idx_loser]})

    # 3. Long flat -- longest hold_days with abs(r_multiple) < 0.3
    flat = trades[trades["r_multiple"].abs() < 0.3]
    if not flat.empty:
        idx_flat = flat["hold_days"].idxmax()
        selections.append({"label": "LONG FLAT", "row": trades.loc[idx_flat]})

    return selections


# ── Bar-by-bar trailing stop replay ──────────────────────────────────────────

def replay_trade(trade_row: pd.Series, d1_df: pd.DataFrame,
                 strategy: Strategy) -> list[dict]:
    """Replay trailing stop logic day-by-day using D1 bars from entry to exit."""
    cfg = strategy.config()
    direction = int(trade_row["direction"])
    entry_price = float(trade_row["entry_price"])
    entry_date = pd.Timestamp(trade_row["entry_date"])
    exit_date = pd.Timestamp(trade_row["exit_date"])

    # Entry bar = D1 bar on entry_date
    if entry_date not in d1_df.index:
        return []

    entry_atr = float(d1_df.loc[entry_date, "atr"])

    stop_dist = cfg.atr_initial_stop * entry_atr
    initial_stop = entry_price - direction * stop_dist

    trailing_stop = initial_stop
    highest_fav = entry_price

    # Get D1 bars from entry to exit (inclusive)
    mask = (d1_df.index >= entry_date) & (d1_df.index <= exit_date)
    bars = d1_df.loc[mask]

    audit_rows = []
    for date, bar in bars.iterrows():
        atr_val = bar.get("atr")
        if pd.isna(atr_val):
            continue

        high = float(bar["high"])
        low = float(bar["low"])
        trail_dist = cfg.atr_trailing_stop * atr_val
        is_entry_day = (date == entry_date)

        if direction == 1:
            if high > highest_fav:
                highest_fav = high
                new_trail = highest_fav - trail_dist
                trailing_stop = max(trailing_stop, new_trail)
            stop_distance = low - trailing_stop
            hit_stop = (not is_entry_day) and (low <= trailing_stop)
        else:
            if low < highest_fav:
                highest_fav = low
                new_trail = highest_fav + trail_dist
                trailing_stop = min(trailing_stop, new_trail)
            stop_distance = trailing_stop - high
            hit_stop = (not is_entry_day) and (high >= trailing_stop)

        audit_rows.append({
            "timestamp": date,
            "open": float(bar["open"]),
            "high": high,
            "low": low,
            "close": float(bar["close"]),
            "atr": float(atr_val),
            "highest_fav": highest_fav,
            "trailing_stop": trailing_stop,
            "dist_to_stop": stop_distance,
            "hit_stop": hit_stop,
        })

        if hit_stop:
            break

    return audit_rows


# ── Format output ────────────────────────────────────────────────────────────

def format_audit(label: str, trade_row: pd.Series, bars: list[dict]) -> str:
    lines = []
    direction_str = "LONG" if int(trade_row["direction"]) == 1 else "SHORT"

    lines.append("=" * 90)
    lines.append(f"  TRADE AUDIT: {label}")
    lines.append("=" * 90)
    lines.append(f"  Instrument:    {trade_row['instrument']}")
    lines.append(f"  Direction:     {direction_str}")
    lines.append(f"  Bucket:        {trade_row['bucket']}")
    lines.append(f"  Entry date:    {trade_row['entry_date']}")
    lines.append(f"  Exit date:     {trade_row['exit_date']}")
    lines.append(f"  Entry price:   {trade_row['entry_price']:.5f}")
    lines.append(f"  Exit price:    {trade_row['exit_price']:.5f}")
    lines.append(f"  Hold days:     {trade_row['hold_days']}")
    lines.append(f"  R-multiple:    {trade_row['r_multiple']:+.3f}")
    lines.append(f"  PnL:           ${trade_row['pnl']:+,.2f}")
    lines.append(f"  Exit reason:   {trade_row['exit_reason']}")
    lines.append("")

    if not bars:
        lines.append("  [No D1 bars found for replay]")
        lines.append("")
        return "\n".join(lines)

    # Initial stop from first bar
    lines.append(f"  Initial stop:  {bars[0]['trailing_stop']:.5f}")
    lines.append(f"  Entry ATR:     {bars[0]['atr']:.5f}")
    lines.append("")

    # Bar-by-bar table
    hdr = (f"  {'Timestamp':>22s}  {'Open':>12s}  {'High':>12s}  {'Low':>12s}  "
           f"{'Close':>12s}  {'ATR':>10s}  {'HighFav':>12s}  {'Trail':>12s}  "
           f"{'Dist':>10s}  {'Hit':>3s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for b in bars:
        hit_marker = ">>>" if b["hit_stop"] else ""
        lines.append(
            f"  {str(b['timestamp']):>22s}  {b['open']:12.5f}  {b['high']:12.5f}  "
            f"{b['low']:12.5f}  {b['close']:12.5f}  {b['atr']:10.5f}  "
            f"{b['highest_fav']:12.5f}  {b['trailing_stop']:12.5f}  "
            f"{b['dist_to_stop']:10.5f}  {hit_marker:>3s}"
        )

    lines.append("")
    lines.append(f"  Total D1 bars replayed: {len(bars)}")
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    strategy_name = sys.argv[1] if len(sys.argv) > 1 else None
    strategy = _find_strategy(strategy_name)
    cfg = strategy.config()
    slug = _slug(cfg.name)

    trades_path = os.path.join("output", slug, "trades.csv")
    # Fallback to old location
    if not os.path.exists(trades_path):
        trades_path = "output/trades.csv"

    print(f"Strategy: {cfg.name}")
    print(f"Loading trades from {trades_path}...")
    trades = load_trades(trades_path)
    if trades.empty:
        print(f"No trades found in {trades_path}")
        return

    print(f"  {len(trades)} trades loaded")

    selections = select_audit_trades(trades)
    print(f"  Selected {len(selections)} trades for audit")

    print("Loading market data & indicators...")
    raw_data = load_all_data()

    # Precompute indicators for needed instruments only
    needed = {s["row"]["instrument"] for s in selections}
    indicators = {}
    for name in needed:
        if name in raw_data:
            indicators[name] = precompute_indicators(
                strategy,
                raw_data[name]["D1"],
            )

    print("\nReplaying trades...\n")
    output_parts = []
    for sel in selections:
        row = sel["row"]
        label = sel["label"]
        inst = row["instrument"]

        if inst not in indicators:
            print(f"  SKIP {label}: no data for {inst}")
            continue

        d1_df = indicators[inst]["D1"]
        bars = replay_trade(row, d1_df, strategy)
        audit_text = format_audit(label, row, bars)
        output_parts.append(audit_text)
        # Print summary
        print(f"  {label:15s} | {inst:8s} | {int(row['direction']):+d} | "
              f"R={row['r_multiple']:+.2f} | {len(bars)} bars replayed")

    full_output = "\n".join(output_parts)
    print("\n" + full_output)

    out_dir = os.path.join("output", slug)
    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, "trade_audit.txt")
    with open(audit_path, "w") as f:
        f.write(full_output)
    print(f"\nSaved: {audit_path}")


if __name__ == "__main__":
    main()
