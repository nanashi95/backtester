#!/usr/bin/env python3
"""
live/signal_runner.py
---------------------
EOD signal runner for the Donchian ensemble live trading system.

Daily workflow
--------------
1. After US equity close (~16:30 ET / 22:30 CET):

       python yf_downloader.py          # refresh all yfinance CSVs
       python live/signal_runner.py     # run EOD signals

2. The report tells you:
   - Which open positions have an updated stop level → update in IBKR
   - Which positions may have stopped out → verify in IBKR
   - What new entries to place at tomorrow's open

3. After confirmed IBKR fills (new entries or stop-outs):
   - Edit live/state.json manually to add/remove positions
   - See state.py docstring for the exact JSON schema

First-time setup
----------------
    python live/signal_runner.py --init 6000 --eurusd 1.08

Options
-------
  --init EUR_AMOUNT    First-time: create state.json with this starting equity
  --eurusd RATE        EUR/USD rate (default 1.05); used for position sizing
  --date YYYY-MM-DD    Override signal date (for replay / testing)
  --no-save            Do not save the report to live/output/
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Data source: yfinance CSVs ─────────────────────────────────────────────────
import data.loader as _loader
import data.yf_data_loader as _yf

_loader.configure(_yf.load_all_data, _yf.get_instrument_bucket)

# ── Live module imports ────────────────────────────────────────────────────────
from live.config import (
    ALL_INSTRUMENTS, SLEEVE_INSTRUMENTS, ACTIVE_SLEEVES,
    RISK_PER_TRADE, GLOBAL_CAP, ATR_INITIAL, ATR_TRAILING,
    ACCOUNT_CURRENCY, make_strategies,
)
from live.state import default_state, load_state, save_state

from engine.signal_engine import SignalEngine, precompute_indicators

# ── Paths ──────────────────────────────────────────────────────────────────────
_LIVE_DIR  = Path(__file__).resolve().parent
STATE_PATH = _LIVE_DIR / "state.json"
OUTPUT_DIR = _LIVE_DIR / "output"

W = 72   # report line width


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dir_str(d: int) -> str:
    return "LONG " if d == 1 else "SHORT"


def _load_indicators(strategies: dict, all_data: dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Precompute Donchian + ATR indicators for every sleeve × instrument."""
    indicators: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sleeve, strat in strategies.items():
        indicators[sleeve] = {}
        for name in SLEEVE_INSTRUMENTS[sleeve]:
            if name not in all_data:
                continue
            d1 = precompute_indicators(strat, all_data[name]["D1"])["D1"]
            indicators[sleeve][name] = d1
    return indicators


def _get_today_bars(indicators: dict, today: pd.Timestamp) -> Dict[str, pd.Series]:
    """
    Return today's indicator row for each instrument.
    Uses the first sleeve that has data for that instrument.
    All sleeves share identical OHLC + ATR(20); only channel periods differ.
    """
    bars: Dict[str, pd.Series] = {}
    for sleeve in ACTIVE_SLEEVES:
        for name, df in indicators[sleeve].items():
            if name not in bars and today in df.index:
                bars[name] = df.loc[today]
    return bars


def _latest_date(indicators: dict) -> Optional[pd.Timestamp]:
    """Return the most recent date present in any indicator dataframe."""
    latest = None
    for sleeve_inds in indicators.values():
        for df in sleeve_inds.values():
            if not df.empty:
                last = df.index[-1]
                if latest is None or last > latest:
                    latest = last
    return latest


def _update_trailing_stops(
    positions: List[dict],
    d1_bars: Dict[str, pd.Series],
) -> List[str]:
    """
    Update highest_fav + trailing_stop for every open position.

    Sets a temporary "_stop_changed" key on each position dict for the report.
    Returns list of human-readable alert strings for positions where today's
    bar crossed the trailing stop (these require IBKR verification).
    """
    alerts: List[str] = []

    for pos in positions:
        bar = d1_bars.get(pos["instrument"])
        if bar is None:
            pos["_stop_changed"] = False
            continue

        atr = bar.get("atr")
        if atr is None or pd.isna(atr):
            pos["_stop_changed"] = False
            continue

        atr        = float(atr)
        trail_dist = ATR_TRAILING * atr
        high       = float(bar["high"])
        low        = float(bar["low"])
        old_stop   = pos["trailing_stop"]

        if pos["direction"] == 1:   # long
            if high > pos["highest_fav"]:
                pos["highest_fav"] = high
                new_trail = pos["highest_fav"] - trail_dist
                if new_trail > pos["trailing_stop"]:
                    pos["trailing_stop"] = new_trail
            if low <= pos["trailing_stop"]:
                alerts.append(
                    f"{pos['instrument']} [{pos['sleeve']}]  "
                    f"low={low:.4f} ≤ stop={pos['trailing_stop']:.4f}"
                )

        else:   # short
            if low < pos["highest_fav"]:
                pos["highest_fav"] = low
                new_trail = pos["highest_fav"] + trail_dist
                if new_trail < pos["trailing_stop"]:
                    pos["trailing_stop"] = new_trail
            if high >= pos["trailing_stop"]:
                alerts.append(
                    f"{pos['instrument']} [{pos['sleeve']}]  "
                    f"high={high:.4f} ≥ stop={pos['trailing_stop']:.4f}"
                )

        pos["_stop_changed"] = (pos["trailing_stop"] != old_stop)

    return alerts


def _generate_signals(
    strategies:  dict,
    indicators:  dict,
    today:       pd.Timestamp,
    state:       dict,
) -> List[dict]:
    """
    Generate new entry signals for today's close.

    Returns a list of signal dicts with an additional "rejected" boolean
    (True = global cap full at time of processing).

    Priority follows ACTIVE_SLEEVES order — first-come, first-served
    matching the backtest execution order.
    """
    open_positions = state["open_positions"]
    total_r_open   = sum(p["r_risked"] for p in open_positions)
    queued_r       = 0.0

    # Convert equity to USD for sizing (account is in EUR)
    equity_usd = state["equity"] * state["eurusd"]

    signals: List[dict] = []

    for sleeve in ACTIVE_SLEEVES:
        insts   = [n for n in SLEEVE_INSTRUMENTS[sleeve] if n in indicators[sleeve]]
        inds    = {n: {"D1": indicators[sleeve][n]} for n in insts}
        sig_eng = SignalEngine(strategies[sleeve], inds, insts)
        raw     = sig_eng.get_signals(today)

        for name, sig in raw.items():
            if sig is None:
                continue

            # Skip if this sleeve already has an open position in this instrument
            if any(
                p["instrument"] == name and p["sleeve"] == sleeve
                for p in open_positions
            ):
                continue

            # Compute theoretical position size
            stop_dist    = ATR_INITIAL * sig.atr
            dollar_risk  = equity_usd * RISK_PER_TRADE
            units_theory = dollar_risk / stop_dist if stop_dist > 0 else 0.0
            est_stop     = sig.d1_close - sig.direction * stop_dist

            entry: Dict[str, Any] = {
                "sleeve":        sleeve,
                "instrument":    name,
                "direction":     sig.direction,
                "atr":           round(sig.atr, 6),
                "signal_date":   str(today.date()),
                "signal_close":  round(sig.d1_close, 6),
                "units_theory":  round(units_theory, 4),
                "est_stop":      round(est_stop, 6),
                "rejected":      False,
                "reject_reason": "",
            }

            # Global cap check (sequential — priority already set by sleeve order)
            if total_r_open + queued_r + 1.0 > GLOBAL_CAP + 1e-9:
                entry["rejected"]      = True
                entry["reject_reason"] = "global_cap"
            else:
                queued_r += 1.0

            signals.append(entry)

    return signals


def _unrealised_pnl(
    positions: List[dict],
    d1_bars:   Dict[str, pd.Series],
) -> float:
    """Return total unrealised P&L in USD across all open positions."""
    total = 0.0
    for pos in positions:
        bar = d1_bars.get(pos["instrument"])
        if bar is not None and pos["units"] > 0:
            close  = float(bar["close"])
            total += (close - pos["entry_price"]) * pos["direction"] * pos["units"]
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(
    today:         pd.Timestamp,
    state:         dict,
    signals:       List[dict],
    stop_alerts:   List[str],
    unrealised_usd: float,
    output_path:   Optional[Path],
) -> None:
    lines: List[str] = []

    def p(s: str = "") -> None:
        lines.append(s)

    eurusd     = state["eurusd"]
    equity_eur = state["equity"]
    equity_usd = equity_eur * eurusd
    nav_usd    = equity_usd + unrealised_usd
    nav_eur    = nav_usd / eurusd
    open_r     = sum(pos["r_risked"] for pos in state["open_positions"])
    sign_u     = "+" if unrealised_usd >= 0 else ""

    # ── Header ────────────────────────────────────────────────────────────────
    p("=" * W)
    p(f"  EOD SIGNAL REPORT — {today.date()}  ({today.strftime('%A')})")
    p(f"  System: ZF-only Donchian ensemble  |  6 sleeves  |  18 instruments")
    p("=" * W)

    # ── Account summary ───────────────────────────────────────────────────────
    p(f"  Equity (realised) : €{equity_eur:>10,.2f}  ≈  ${equity_usd:>10,.2f}")
    p(f"  Unrealised P&L    : {sign_u}${unrealised_usd:>9,.2f}  ≈  {sign_u}€{unrealised_usd/eurusd:>9,.2f}")
    p(f"  NAV               :  ${nav_usd:>9,.2f}  ≈   €{nav_eur:>9,.2f}")
    p(f"  Open R            :  {open_r:.1f}R  /  {GLOBAL_CAP:.0f}R")
    p(f"  EUR/USD           :  {eurusd:.4f}")

    # ── Open positions ────────────────────────────────────────────────────────
    positions = state["open_positions"]
    p("")
    p("─" * W)
    p(f"  OPEN POSITIONS — {len(positions)}")

    if not positions:
        p("  (none)")
    else:
        p(f"  {'Sleeve':<8} {'Instrument':<12} {'Dir':<6} {'Entry Date':<12}"
          f" {'Entry':>10} {'Stop':>10}  {'':8} {'Days':>4}")
        p(f"  {'─' * 70}")
        for pos in positions:
            entry_ts   = pd.Timestamp(pos["entry_date"])
            days_held  = (today - entry_ts).days
            changed    = "↑ UPDATE" if pos.get("_stop_changed") else ""
            p(f"  {pos['sleeve']:<8} {pos['instrument']:<12} "
              f"{_dir_str(pos['direction']):<6} {pos['entry_date']:<12}"
              f" {pos['entry_price']:>10.4f} {pos['trailing_stop']:>10.4f}"
              f"  {changed:<8} {days_held:>3}d")
        p("")
        p("  ↑ UPDATE  =  stop level raised today  —  cancel old stop in IBKR,"
          " place new one")

    # ── Stop alerts ───────────────────────────────────────────────────────────
    p("")
    p("─" * W)
    p("  ⚠  POTENTIAL STOP HITS — bar crossed stop level today")
    if not stop_alerts:
        p("  (none)")
    else:
        p("  Verify each fill in IBKR, then remove position from state.json:")
        for alert in stop_alerts:
            p(f"  !!!  {alert}")

    # ── New signals ───────────────────────────────────────────────────────────
    accepted = [s for s in signals if not s["rejected"]]
    rejected = [s for s in signals if s["rejected"]]

    p("")
    p("─" * W)
    tomorrow = pd.Timestamp(today) + pd.tseries.offsets.BDay(1)
    p(f"  ENTRY SIGNALS — place at OPEN on {tomorrow.date()}  ({len(accepted)} accepted)")

    if not accepted:
        p("  (none)")
    else:
        p(f"  {'Sleeve':<8} {'Instr':<8} {'Dir':<6} {'Close':>10}"
          f" {'ATR':>8} {'Est.Stop':>10} {'Theory Units':>13}")
        p(f"  {'─' * 68}")
        for s in accepted:
            p(f"  {s['sleeve']:<8} {s['instrument']:<8} {_dir_str(s['direction']):<6}"
              f" {s['signal_close']:>10.4f} {s['atr']:>8.4f}"
              f" {s['est_stop']:>10.4f} {s['units_theory']:>10.2f}")
        p("")
        p("  Theory units < 1 = account too small for 1 contract at 0.6% risk.")
        p("  For paper trading: place 1 contract regardless (risk % will be higher).")

    if rejected:
        p("")
        rej_str = "  ".join(
            f"{s['instrument']} [{s['sleeve']}]" for s in rejected
        )
        p(f"  REJECTED (cap full at {GLOBAL_CAP:.0f}R): {rej_str}")

    # ── Footer ────────────────────────────────────────────────────────────────
    p("")
    if state.get("last_run"):
        p(f"  Last state saved : {state['last_run'][:19]} UTC")
    p("=" * W)

    report = "\n".join(lines)
    print(report)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\n  Report saved: {output_path.relative_to(_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_init(initial_equity_eur: float, eurusd: float) -> None:
    if STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} already exists.")
        print("  Delete it first, or edit it manually to reset equity/positions.")
        sys.exit(1)
    state = default_state(initial_equity_eur, eurusd)
    save_state(STATE_PATH, state)
    print(f"  Initialised:  equity = €{initial_equity_eur:,.2f}  "
          f"EUR/USD = {eurusd:.4f}")
    print(f"  State file :  {STATE_PATH.relative_to(_ROOT)}")
    print()
    print("  Next step: refresh yfinance data and run the EOD signal runner:")
    print("    python yf_downloader.py")
    print("    python live/signal_runner.py")


def cmd_run(date_override: Optional[str], save_report: bool) -> None:
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found.")
        print("  Run first-time setup:  python live/signal_runner.py --init 6000")
        sys.exit(1)

    state      = load_state(STATE_PATH)
    strategies = make_strategies()

    # ── Load data ─────────────────────────────────────────────────────────────
    warmup   = "2007-01-01"
    end_date = date_override or date.today().isoformat()
    print(f"\n  Loading historical data (start={warmup}) ...")
    all_data = _yf.load_all_data(start=warmup, end=end_date)

    # ── Precompute indicators ─────────────────────────────────────────────────
    print(f"  Precomputing indicators ({len(ACTIVE_SLEEVES)} sleeves × "
          f"{len(ALL_INSTRUMENTS)} instruments) ...")
    indicators = _load_indicators(strategies, all_data)

    # ── Determine signal date ─────────────────────────────────────────────────
    today = _latest_date(indicators)
    if today is None:
        print("ERROR: no indicator data available — check yfinance CSVs.")
        sys.exit(1)
    if date_override:
        today = pd.Timestamp(date_override)

    print(f"  Signal date : {today.date()}\n")

    # ── Today's bars ──────────────────────────────────────────────────────────
    d1_bars = _get_today_bars(indicators, today)

    # ── Update trailing stops ─────────────────────────────────────────────────
    stop_alerts = _update_trailing_stops(state["open_positions"], d1_bars)

    # ── Generate entry signals ────────────────────────────────────────────────
    signals = _generate_signals(strategies, indicators, today, state)

    # ── Update pending_signals in state (accepted only; remove internal keys) ─
    state["pending_signals"] = [
        {k: v for k, v in s.items()
         if k not in ("rejected", "reject_reason")}
        for s in signals
        if not s["rejected"]
    ]

    # ── Unrealised P&L (display only — not persisted) ─────────────────────────
    unrealised_usd = _unrealised_pnl(state["open_positions"], d1_bars)

    # ── Save state (strip temporary report flags before writing) ──────────────
    for pos in state["open_positions"]:
        pos.pop("_stop_changed", None)
    save_state(STATE_PATH, state)

    # ── Output ────────────────────────────────────────────────────────────────
    output_path = (
        OUTPUT_DIR / f"report_{today.date()}.txt"
        if save_report else None
    )
    _print_report(today, state, signals, stop_alerts, unrealised_usd, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOD signal runner — Donchian ensemble live system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--init", metavar="EUR", type=float,
        help="First-time setup: create state.json with this starting equity (EUR).",
    )
    parser.add_argument(
        "--eurusd", type=float, default=1.05,
        help="EUR/USD rate for position sizing (default 1.05).",
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Override signal date (for replay / testing).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving the text report to live/output/.",
    )
    args = parser.parse_args()

    if args.init is not None:
        cmd_init(args.init, args.eurusd)
    else:
        cmd_run(
            date_override = args.date,
            save_report   = not args.no_save,
        )


if __name__ == "__main__":
    main()
