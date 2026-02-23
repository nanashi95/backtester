"""
loader.py
---------
Swappable data source abstraction.

Engines import load_all_data / get_instrument_bucket from here instead of
directly from mt5_data_loader.  Default behaviour is identical to MT5
(zero-config for existing code).  Call configure() once before running to
switch to IBKR or any other source.

Usage (IBKR path):
    from data import loader
    import data.ibkr_data_loader as ibkr_loader
    loader.configure(ibkr_loader.load_all_data, ibkr_loader.get_instrument_bucket)
"""

_load_fn   = None
_bucket_fn = None


def configure(load_fn, bucket_fn):
    """Point the loader at a different data source.

    Args:
        load_fn:   callable(start: str, end: str) -> Dict[str, Dict[str, DataFrame]]
        bucket_fn: callable(name: str) -> str
    """
    global _load_fn, _bucket_fn
    _load_fn, _bucket_fn = load_fn, bucket_fn


def load_all_data(start: str, end: str):
    """Load OHLC data for all instruments in [start, end].

    Returns {instrument_name: {"D1": df, "H4": df_or_None}}.
    """
    if _load_fn is None:
        from data.mt5_data_loader import load_all_data as _mt5_load
        return _mt5_load(start, end)
    return _load_fn(start, end)


def get_instrument_bucket(name: str) -> str:
    """Return the portfolio bucket for an instrument (e.g. 'metals', 'rates')."""
    if _bucket_fn is None:
        from data.mt5_data_loader import BUCKET_MAP
        return BUCKET_MAP.get(name, "unknown")
    return _bucket_fn(name)
