"""
live/config.py
--------------
Production system configuration — locked baseline 2026-02-24.

Matches zf_risk06_backtest.py exactly:
  Universe : 18 instruments (ZF + 4 equity + 5 metals + 2 energy + 6 agri)
  Risk     : 0.6% / trade  |  1.5×ATR initial  |  2.0×ATR trailing  |  6R cap
  Results  : CAGR +9.39%  MaxDD -25.59%  MAR 0.367  Sharpe 0.55  (2008–2025)

DO NOT change these constants without running a full backtest to verify the new
configuration produces equal or better results.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Project root (safe to import from any working directory) ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategies.donchian_ensemble import make_sleeve

# ── Universe ───────────────────────────────────────────────────────────────────

RATES       = ["ZF"]
EQUITY      = ["ES", "NQ", "RTY", "YM"]
METALS      = ["GC", "SI", "HG", "PA", "PL"]
ENERGY      = ["CL", "NG"]
AGRICULTURE = ["ZW", "ZS", "ZC", "SB", "KC", "CC"]

ALL_INSTRUMENTS = RATES + EQUITY + METALS + ENERGY + AGRICULTURE  # 18 total

SLEEVE_INSTRUMENTS = {
    "B_ME": METALS + ENERGY,
    "C_ME": METALS + ENERGY,
    "B_EA": EQUITY + AGRICULTURE,
    "C_EA": EQUITY + AGRICULTURE,
    "B_R":  RATES,
    "C_R":  RATES,
}

# ── Risk ───────────────────────────────────────────────────────────────────────

RISK_PER_TRADE = 0.006   # 0.6% of equity per trade (1R)
GLOBAL_CAP     = 6.0     # max total R open across all sleeves simultaneously
ATR_INITIAL    = 1.5     # initial stop = 1.5 × ATR(20) from entry price
ATR_TRAILING   = 2.0     # trailing stop = 2.0 × ATR(20), ratchets in fav direction

# ── Sleeves ────────────────────────────────────────────────────────────────────
# Speed-differentiated: B = faster breakout period, C = slower.
# Periods chosen to match the locked backtest configuration exactly.

SLEEVE_PERIODS = {
    "B_ME": (50,  "B"),   # Metals + Energy  — medium breakout
    "C_ME": (100, "C"),   # Metals + Energy  — slow breakout
    "B_EA": (75,  "B"),   # Equity + Agri    — medium breakout
    "C_EA": (150, "C"),   # Equity + Agri    — slow breakout
    "B_R":  (50,  "B"),   # Rates            — medium breakout
    "C_R":  (100, "C"),   # Rates            — slow breakout
}

ACTIVE_SLEEVES = list(SLEEVE_PERIODS.keys())


def make_strategies() -> dict:
    """Return fresh strategy instances for all 6 sleeves."""
    return {
        name: make_sleeve(period, label)
        for name, (period, label) in SLEEVE_PERIODS.items()
    }


# ── IBKR connection ────────────────────────────────────────────────────────────

IB_HOST       = "127.0.0.1"
IB_PORT_PAPER = 4002     # IB Gateway (paper trading)
IB_PORT_LIVE  = 4001     # IB Gateway (live)
IB_CLIENT_ID  = 20       # client ID for signal runner process

# ── Account ────────────────────────────────────────────────────────────────────

ACCOUNT_CURRENCY = "EUR"   # display currency; instruments are USD-denominated
