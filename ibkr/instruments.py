"""
ibkr/instruments.py
-------------------
Target universe for the IBKR futures-based backtest.

Each entry maps an internal name to a tuple:
    (ibkr_symbol, exchange, currency, secType, bucket)

secType = 'CONTFUT'  — continuous front-month contract, best for backtesting
                       (avoids roll gaps; IB splices adjacent contracts).

Bucket names align with ensemble_engine / risk_engine conventions.
"""

INSTRUMENTS = {
    # ── Rates (primary addition vs MT5) ──────────────────────────────────────
    "ZN":  ("ZN",  "CBOT",  "USD", "CONTFUT", "rates"),   # 10Y Treasury Note
    "ZB":  ("ZB",  "CBOT",  "USD", "CONTFUT", "rates"),   # 30Y Treasury Bond
    "ZF":  ("ZF",  "CBOT",  "USD", "CONTFUT", "rates"),   # 5Y Treasury Note

    # ── Equity indices ────────────────────────────────────────────────────────
    "ES":  ("ES",  "CME",   "USD", "CONTFUT", "equity"),  # S&P 500 E-mini
    "NQ":  ("NQ",  "CME",   "USD", "CONTFUT", "equity"),  # Nasdaq E-mini
    "RTY": ("RTY", "CME",   "USD", "CONTFUT", "equity"),  # Russell 2000
    "YM":  ("YM",  "CBOT",  "USD", "CONTFUT", "equity"),  # Dow E-mini

    # ── FX futures ────────────────────────────────────────────────────────────
    "6J":  ("6J",  "CME",   "USD", "CONTFUT", "fx"),      # JPY/USD
    "6E":  ("6E",  "CME",   "USD", "CONTFUT", "fx"),      # EUR/USD
    "6A":  ("6A",  "CME",   "USD", "CONTFUT", "fx"),      # AUD/USD
    "6C":  ("6C",  "CME",   "USD", "CONTFUT", "fx"),      # CAD/USD
    "6B":  ("6B",  "CME",   "USD", "CONTFUT", "fx"),      # GBP/USD

    # ── Metals ───────────────────────────────────────────────────────────────
    "GC":  ("GC",  "COMEX", "USD", "CONTFUT", "metals"),  # Gold
    "SI":  ("SI",  "COMEX", "USD", "CONTFUT", "metals"),  # Silver
    "HG":  ("HG",  "COMEX", "USD", "CONTFUT", "metals"),  # Copper
    "PA":  ("PA",  "NYMEX", "USD", "CONTFUT", "metals"),  # Palladium
    "PL":  ("PL",  "NYMEX", "USD", "CONTFUT", "metals"),  # Platinum

    # ── Energy (included for testing — likely excluded after results) ─────────
    "CL":  ("CL",  "NYMEX", "USD", "CONTFUT", "energy"),  # Crude Oil
    "NG":  ("NG",  "NYMEX", "USD", "CONTFUT", "energy"),  # Natural Gas

    # ── Agriculture ───────────────────────────────────────────────────────────
    "ZW":  ("ZW",  "CBOT",  "USD", "CONTFUT", "agriculture"),  # Wheat
    "ZS":  ("ZS",  "CBOT",  "USD", "CONTFUT", "agriculture"),  # Soybean
    "ZC":  ("ZC",  "CBOT",  "USD", "CONTFUT", "agriculture"),  # Corn
    "SB":  ("SB",  "NYBOT", "USD", "CONTFUT", "agriculture"),  # Sugar
    "KC":  ("KC",  "NYBOT", "USD", "CONTFUT", "agriculture"),  # Coffee
    "CC":  ("CC",  "NYBOT", "USD", "CONTFUT", "agriculture"),  # Cocoa
}

# Flat bucket lookup — used by ibkr_data_loader.get_instrument_bucket()
BUCKET_MAP = {name: spec[4] for name, spec in INSTRUMENTS.items()}

# Ordered list of all instrument names
INSTRUMENT_NAMES = list(INSTRUMENTS.keys())
