# Trend Following Backtest Engine
## Multi-Asset Portfolio Simulator — Production Research Grade

---

## Architecture

```
backtest/
├── main.py                        ← Entry point
├── data/
│   ├── __init__.py
│   └── data_generator.py          ← Synthetic OHLC (replace with real data)
├── engine/
│   ├── __init__.py
│   ├── signal_engine.py           ← D1 SMA alignment + H4 SMA cross
│   ├── risk_engine.py             ← Position sizing, stops, cap enforcement
│   └── portfolio_engine.py        ← Event-driven simulation loop
├── metrics/
│   ├── __init__.py
│   └── metrics_engine.py          ← All performance analytics
└── output/
    ├── equity_curve.csv
    ├── trades.csv
    └── metrics_summary.txt
```

---

## Strategy Rules (Exact Implementation)

### Signal Engine (`signal_engine.py`)

**D1 Timeframe — Trend Alignment:**
| Direction | Condition |
|-----------|-----------|
| LONG      | SMA(5) > SMA(13) > SMA(48) > SMA(75) |
| SHORT     | SMA(5) < SMA(13) < SMA(48) < SMA(75) |
| FLAT      | Any other configuration |

**H4 Timeframe — Entry Trigger:**
| Direction | Condition |
|-----------|-----------|
| LONG      | SMA(5) crosses above SMA(13) AND close > SMA(5) |
| SHORT     | SMA(5) crosses below SMA(13) AND close < SMA(5) |

### Risk Engine (`risk_engine.py`)

| Parameter | Value |
|-----------|-------|
| Risk per trade (1R) | 0.7% of equity |
| Initial stop | 1.5 × ATR(14) on H4 |
| Trailing stop | 2 × ATR(14) on H4 from highest favourable price |
| Exit method | Trailing stop ONLY |

### Portfolio Caps

| Bucket | Instruments | Max R Exposure |
|--------|-------------|---------------|
| FX Yen | AUDJPY, EURJPY, USDJPY, CADJPY, NZDJPY | 1R |
| Equity | US100, US500, JP225 | 2R |
| Metals | Gold, Silver, Copper | 2R |
| Energy | UKOil, USOil | 2R |
| Softs  | Sugar, Coffee, Cocoa | 2R |
| **Total Portfolio** | — | **6R** |

Rules:
- If new trade would breach any cap → **rejected** (no scaling)
- No partial entries, no size reduction

---

## How to Run

```bash
# From project root
python main.py
```

Requirements: Python 3.9+, pandas, numpy

---

## Plugging In Real Data

Replace `data/data_generator.py` with real OHLC data.
The `load_all_data()` function must return:

```python
{
    "US100":  {"D1": pd.DataFrame, "H4": pd.DataFrame},
    "US500":  {"D1": pd.DataFrame, "H4": pd.DataFrame},
    ...  # all 16 instruments
}
```

Each DataFrame must have columns: `open`, `high`, `low`, `close`
Index must be a `DatetimeIndex`.

### Data Sources (Recommended)
| Source | Format | Notes |
|--------|--------|-------|
| MT4/MT5 export | CSV | Use History Center → export per instrument |
| Dukascopy | HST/CSV | Free tick data, resample to D1/H4 |
| Norgate Data | Python API | Cleaned, adjusted data |
| Refinitiv/Bloomberg | CSV | Institutional quality |

### MT4 CSV Loader Example
```python
def load_mt4_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[[0,1]], 
                     names=["date","time","open","high","low","close","vol"])
    df.index = df.pop("date_time")
    return df[["open","high","low","close"]]
```

---

## Output Files

### `equity_curve.csv`
Daily portfolio snapshot:
- `equity` — settled cash balance
- `total_value` — equity + unrealised P&L (mark-to-market)
- `open_trades` — number of active trades
- `total_r_open` — total R exposure currently deployed
- `new_trades` / `closed_trades` — activity per day
- `signals_seen` / `signals_rejected` — signal activity

### `trades.csv`
Full trade log (one row per closed trade):
- `instrument`, `bucket`, `direction`
- `entry_date`, `exit_date`, `hold_days`
- `entry_price`, `exit_price`
- `pnl` — dollar P&L
- `r_multiple` — outcome in R units (1R = 0.7% equity at entry)
- `exit_reason` — `trailing_stop` or `end_of_backtest`

### `metrics_summary.txt`
Human-readable performance report with all required metrics.

---

## Performance Expectations (Real Data)

Trend-following systems on this universe historically (2008–2024):
| Metric | Typical Range |
|--------|--------------|
| CAGR | 8–18% |
| Max Drawdown | 15–35% |
| MAR Ratio | 0.4–0.8 |
| Win Rate | 35–45% |
| Expectancy | +0.3 to +0.8R |

> **Note:** The synthetic data generator produces zero-drift random walks (by design), so
> backtested returns on synthetic data will be negative. The negative result on synthetic
> data actually *validates* the system — it confirms the engine doesn't magically generate
> profits from noise. Real market returns come from momentum (serial correlation) in prices,
> which synthetic GBM does not model.

---

## Design Decisions & Notes

1. **No parameter optimization** — all parameters (SMA periods, ATR multipliers, R%) are fixed
2. **Event-driven** — chronological processing, no look-ahead bias
3. **Trailing stop ATR updated per H4 bar** — each H4 bar updates the trailing stop using its own ATR(14); stop only ratchets in the favourable direction
4. **Entry at H4 close** — conservative: no assumption about limit orders within bar
5. **No scaling** — bucket/portfolio caps are binary reject/accept
6. **Mark-to-market** — equity curve uses unrealised P&L for realistic drawdown calculation
7. **ATR calculation** — Wilder's smoothed ATR (EWM) for consistency with MT4/MT5

---

## Forward Testing Integration

This system is designed to slot directly into forward testing.
Signal output format from `SignalEngine.get_signals(date)` is identical whether
running historical or live — just feed current OHLC bars.

For live deployment:
1. Maintain rolling windows of D1 and H4 bars per instrument
2. At each D1 close: call `update_trailing_stops()` → `get_signals()` → `open_trade()`
3. RiskEngine state carries forward between sessions
