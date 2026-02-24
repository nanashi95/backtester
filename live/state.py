"""
live/state.py
-------------
Persist and load live trading state across daily EOD runs.

State file: live/state.json

Schema
------
{
  "version":         1,
  "initial_equity":  float,   # starting equity in account currency (EUR)
  "equity":          float,   # realised equity (closed P&L accumulated since start)
  "eurusd":          float,   # EUR/USD rate used for position sizing
  "open_positions":  [...],   # see Open Position schema below
  "pending_signals": [...],   # signals queued for next-day open
  "trade_counter":   int,     # monotonic trade ID counter
  "last_run":        str      # ISO-8601 datetime of last successful EOD run
}

Open Position schema
--------------------
{
  "trade_id":            int,
  "sleeve":              str,    # "B_ME", "C_ME", "B_EA", "C_EA", "B_R", "C_R"
  "instrument":          str,    # "ZF", "GC", "ES", etc.
  "direction":           int,    # +1 long, -1 short
  "entry_date":          str,    # "YYYY-MM-DD"
  "entry_price":         float,
  "units":               float,  # actual contracts/units in the position
  "initial_stop":        float,
  "trailing_stop":       float,  # current stop — updated each EOD
  "highest_fav":         float,  # most favourable extreme seen (for trailing calc)
  "r_risked":            float,  # 1.0 for standard position
  "entry_equity":        float,  # equity (EUR) at trade entry — for R calc on close
  "ibkr_stop_order_id":  int     # IBKR order ID of the active stop order; 0 = none
}

Pending Signal schema
---------------------
{
  "sleeve":        str,
  "instrument":    str,
  "direction":     int,    # +1 or -1
  "atr":           float,
  "signal_date":   str,    # date signal fired
  "signal_close":  float,  # close price on signal day
  "units_theory":  float,  # sizing formula result (may be < 1 at small equity)
  "est_stop":      float   # estimated stop price (signal_close ± 1.5×ATR)
}

Manual state management (Phase 1 — before auto-reconciler)
-----------------------------------------------------------
After a confirmed IBKR fill, add a position by appending to open_positions:

  {
    "trade_id":           <next trade_counter + 1>,
    "sleeve":             "B_ME",
    "instrument":         "GC",
    "direction":          1,
    "entry_date":         "2026-02-25",
    "entry_price":        2950.00,
    "units":              1,
    "initial_stop":       2900.00,
    "trailing_stop":      2900.00,
    "highest_fav":        2950.00,
    "r_risked":           1.0,
    "entry_equity":       6000.0,
    "ibkr_stop_order_id": 0
  }

Also increment "trade_counter" by 1.

After a stop fill, remove the position from open_positions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

STATE_VERSION = 1


def default_state(initial_equity_eur: float, eurusd: float = 1.05) -> Dict[str, Any]:
    """Return a fresh empty state for first-time initialisation."""
    return {
        "version":         STATE_VERSION,
        "initial_equity":  initial_equity_eur,
        "equity":          initial_equity_eur,
        "eurusd":          eurusd,
        "open_positions":  [],
        "pending_signals": [],
        "trade_counter":   0,
        "last_run":        None,
    }


def load_state(path: Path) -> Dict[str, Any]:
    """Load state from JSON. Raises FileNotFoundError if not found."""
    with open(path) as f:
        return json.load(f)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """Atomically save state to JSON (write → rename)."""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(path)


def get_bucket(instrument: str) -> str:
    """Return portfolio bucket for an instrument (e.g. 'metals', 'rates')."""
    from engine.risk_engine import BUCKET_MAP
    return BUCKET_MAP.get(instrument, "unknown")
