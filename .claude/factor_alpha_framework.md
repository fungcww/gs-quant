# Factor-Neutral Alpha Framework (HK) — Rules & Conventions

## Module Progression
| Module | File | Goal |
|--------|------|------|
| M5.1 | `beta_engine.py` | Rolling OLS beta vs HSTECH; residual (idiosyncratic) returns for 1810.HK |
| M5.2 | `tournament_optimizer.py` | TechnicalOnly vs SentimentAugmented vs MarketNeutralAlpha comparison engine |
| M5.3 | `vol_sizer_hk.py` | ATR/rolling-std position sizing + HKEX lot-size constraint + 5 bps slippage |
| M5.4 | `lead_lag_monitor.py` | Overnight US Tech (QQQ/TSLA) → 1810.HK open spillover factor |

---

## HK Asset Conventions

### Tickers (yfinance)
- **Target**: `1810.HK` — Xiaomi Corp, listed on HKEX
- **Benchmark**: `3033.HK` — CSOP Hang Seng TECH Index ETF (yfinance proxy for HSTECH; `^HSTECH` is not available on yfinance)
- **US Lead-Lag anchor (M5.4)**: `QQQ` or `TSLA` (already in market.db)

### HKEX Lot Size Rule (critical for M5.3)
- **1810.HK minimum board lot**: 200 shares
- All position sizes **must be multiples of 200** — round down to nearest lot
- Example: calculated size = 730 shares → actual order = 600 shares (3 lots)
- Verify the current lot size before each module release (HKEX can change it)

### Currency
- Prices stored in **HKD** as-is from yfinance
- Returns are dimensionless (%) — no FX conversion needed for signal/sizing logic
- If USD comparison is ever needed, use a fixed approximation or fetch USD/HKD

### Trading Hours (HKT = UTC+8)
- Morning session: 09:30–12:00
- Afternoon session: 13:00–16:00
- Overnight gap vs US close: ~16 hours (US close 16:00 ET → HK open 09:30 HKT next day)

### Execution Costs (M5.3 baseline)
- Slippage: **5 basis points (0.05%)** per side — HK market liquidity cost
- Brokerage: apply Futu HK stock fee structure (separate from US OrderCost — check Futu HK rate card)

---

## Data Storage
- HK tickers stored in the **same `market.db`** as US tickers (same schema)
- `ensure_data()` in each module handles idempotent yfinance download + INSERT OR REPLACE
- Date range: **2024-01-01 to 2025-12-31** (aligned with US backtest window)

---

## Chart Output Rule
All PNGs produced by any HK module must be saved to **`charts/`** (project root), never to root or subfolders. Follow same convention as US backtest runner.

---

## gs-quant Library-First Rule (applies here too)
- **Beta**: use `gs_quant.timeseries.econometrics.beta()` — do not reimplement OLS
- **Volatility**: use `gs_quant.timeseries.econometrics.volatility()` or `statistics.std()`
- **Correlation**: use `gs_quant.timeseries.econometrics.correlation()`
- Only write custom numpy/pandas math when gs-quant has no equivalent
