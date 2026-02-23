"""
ibkr/instruments.py
-------------------
Target universe for the IBKR futures-based backtest.

Each entry maps an internal name to a tuple:
    (ibkr_symbol, exchange, currency, secType, bucket, invert)

secType = 'CONTFUT'  — continuous front-month futures (commodities, rates, equity)
secType = 'CASH'     — FX spot via IDEALPRO (same Donchian signals as FX futures)

invert = True  — store 1/price in CSV, swapping H/L (needed when IBKR quotes the
                 pair in the opposite direction to the futures contract)
                 6J: futures = JPY/USD direction → download USD/JPY → invert
                 6C: futures = CAD/USD direction → download USD/CAD → invert

whatToShow:
  CONTFUT instruments use 'TRADES' (last sale price, standard for futures)
  CASH instruments use 'MIDPOINT' (bid/ask mid, standard for FX spot)
"""

INSTRUMENTS = {
    # ── Rates ─────────────────────────────────────────────────────────────────
    #                  symbol  exchange  currency  secType   bucket   invert
    "ZN":  ("ZN",  "CBOT",     "USD", "CONTFUT", "rates",   False),
    "ZB":  ("ZB",  "CBOT",     "USD", "CONTFUT", "rates",   False),
    "ZF":  ("ZF",  "CBOT",     "USD", "CONTFUT", "rates",   False),

    # ── Equity indices ────────────────────────────────────────────────────────
    "ES":  ("ES",  "CME",      "USD", "CONTFUT", "equity",  False),
    "NQ":  ("NQ",  "CME",      "USD", "CONTFUT", "equity",  False),
    "RTY": ("RTY", "CME",      "USD", "CONTFUT", "equity",  False),
    "YM":  ("YM",  "CBOT",     "USD", "CONTFUT", "equity",  False),

    # ── FX (spot via IDEALPRO — Donchian signals identical to FX futures) ─────
    # 6E, 6A, 6B: IBKR quotes in the same direction as the futures contract
    "6E":  ("EUR", "IDEALPRO", "USD", "CASH",    "fx",      False),  # EUR/USD
    "6A":  ("AUD", "IDEALPRO", "USD", "CASH",    "fx",      False),  # AUD/USD
    "6B":  ("GBP", "IDEALPRO", "USD", "CASH",    "fx",      False),  # GBP/USD
    # 6J, 6C: IBKR quotes USD as base → invert to match futures direction
    "6J":  ("USD", "IDEALPRO", "JPY", "CASH",    "fx",      True),   # 1/(USD/JPY) → JPY/USD
    "6C":  ("USD", "IDEALPRO", "CAD", "CASH",    "fx",      True),   # 1/(USD/CAD) → CAD/USD

    # ── Metals ───────────────────────────────────────────────────────────────
    "GC":  ("GC",  "COMEX",    "USD", "CONTFUT", "metals",  False),
    "SI":  ("SI",  "COMEX",    "USD", "CONTFUT", "metals",  False),
    "HG":  ("HG",  "COMEX",    "USD", "CONTFUT", "metals",  False),
    "PA":  ("PA",  "NYMEX",    "USD", "CONTFUT", "metals",  False),
    "PL":  ("PL",  "NYMEX",    "USD", "CONTFUT", "metals",  False),

    # ── Energy ────────────────────────────────────────────────────────────────
    "CL":  ("CL",  "NYMEX",    "USD", "CONTFUT", "energy",  False),
    "NG":  ("NG",  "NYMEX",    "USD", "CONTFUT", "energy",  False),

    # ── Agriculture ───────────────────────────────────────────────────────────
    "ZW":  ("ZW",  "CBOT",     "USD", "CONTFUT", "agriculture", False),
    "ZS":  ("ZS",  "CBOT",     "USD", "CONTFUT", "agriculture", False),
    "ZC":  ("ZC",  "CBOT",     "USD", "CONTFUT", "agriculture", False),
    "SB":  ("SB",  "NYBOT",    "USD", "CONTFUT", "agriculture", False),
    "KC":  ("KC",  "NYBOT",    "USD", "CONTFUT", "agriculture", False),
    "CC":  ("CC",  "NYBOT",    "USD", "CONTFUT", "agriculture", False),
}

# Flat bucket lookup — used by ibkr_data_loader.get_instrument_bucket()
BUCKET_MAP = {name: spec[4] for name, spec in INSTRUMENTS.items()}

# Ordered list of all instrument names
INSTRUMENT_NAMES = list(INSTRUMENTS.keys())
