# Strategy Rules (as of M3.2)

## Universe
- 10 tickers: AAPL, SMR, NVDA, TSLA, GOOGL, MSFT, META, AMD, NNE, OKLO
- Market data: daily OHLC from yfinance, persisted in `market.db` (parallel download via ThreadPoolExecutor)
- Data split: 2024 in-sample, 2025 out-of-sample

## Portfolio Correlation Engine (M3.1 / M3.2)
- Pearson correlation matrix of 2024 in-sample daily returns printed at run start
- **Dynamic Buy Gate**: new long entry blocked if 30-day rolling Pearson correlation between the candidate ticker's returns and any currently-held ticker's returns is ≥ threshold
- M3.1 threshold: 0.70 | **M3.2 retail threshold: 0.45** (tighter — small account cannot afford two holdings moving together against it)
- Prevents sector-concentration flush (AI cluster: NVDA/MSFT/META/AMD; Nuclear: SMR/NNE/OKLO)
- Implemented in `MACrossoverEODTrigger._rolling_corr_gate`

## Regime Detection
- **Trend mode** (ADX > threshold, default 25): SMA crossover entry; price < SMA exit
- **Mean reversion mode** (ADX < 20): RSI(14) < 30 + Lower Bollinger Band touch entry; Upper Bollinger Band exit
- ADX hysteresis zone (20–25): no new entries, hold existing positions

## Position Sizing & Risk (M3.2 — Retail Inverse-Vol, Default)
- **Sizing**: Inverse-volatility via 20-day rolling std of daily % returns (numpy-backed)
  - `risk_usd = 0.5% × current NAV`
  - `shares = risk_usd / (fill_ref_price × 20d_vol)` — fractional shares allowed
  - Per-position cap: max 50% of NAV to prevent over-sizing in calm periods
- **Stop loss**: ATR Chandelier trailing stop (preserved from Stage 7; active alongside vol sizing)
- **Initial equity**: $10,000 default for Stage 32 (pass `--initial-value` to override)
- Trade notional printed at entry: `[M3.2] TICKER BUY X.XXXX sh × $P = $N notional | 20d-vol=V | risk=$R`

## Position Sizing & Risk (M3.1 legacy — ATR-based)
- Sizing: ATR-based integer shares (Stage 7 path, `--stage 31`)
- Stop loss: Chandelier stop (preserved in both regime modes)

## Execution Model (M2.1)
- Signal generated at Close of Day T (EOD trigger at 23:00)
- Fill executes at Open of Day T+1 (`_fill_ref = next_day_open`, slippage applied to open price)
- Position sizing uses `_fill_ref` (next-day open) as the share-count basis
- Fallback to Close T if next-day open is unavailable (last bar / missing data)

## Execution Costs
- Slippage: 0.05% applied to fill reference price
- Fees: Futu-style US stock estimate (via `OrderCost`)
