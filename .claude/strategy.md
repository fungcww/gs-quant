# Strategy Rules (as of M3.1)

## Universe
- 10 tickers: AAPL, SMR, NVDA, TSLA, GOOGL, MSFT, META, AMD, NNE, OKLO
- Market data: daily OHLC from yfinance, persisted in `market.db` (parallel download via ThreadPoolExecutor)
- Data split: 2024 in-sample, 2025 out-of-sample

## Portfolio Correlation Engine (M3.1)
- Pearson correlation matrix of 2024 in-sample daily returns printed at run start
- **Dynamic Buy Gate**: new long entry blocked if 30-day rolling Pearson correlation between the candidate ticker's returns and any currently-held ticker's returns is ≥ 0.70
- Prevents sector-concentration flush (AI cluster: NVDA/MSFT/META/AMD; Nuclear: SMR/NNE/OKLO)
- Implemented in `MACrossoverEODTrigger._rolling_corr_gate`

## Regime Detection
- **Trend mode** (ADX > threshold, default 25): SMA crossover entry; price < SMA exit
- **Mean reversion mode** (ADX < 20): RSI(14) < 30 + Lower Bollinger Band touch entry; Upper Bollinger Band exit
- ADX hysteresis zone (20–25): no new entries, hold existing positions

## Position Sizing & Risk
- Sizing: ATR-based
- Stop loss: Chandelier stop (preserved in both regime modes)

## Execution Model (M2.1)
- Signal generated at Close of Day T (EOD trigger at 23:00)
- Fill executes at Open of Day T+1 (`_fill_ref = next_day_open`, slippage applied to open price)
- Position sizing uses `_fill_ref` (next-day open) as the share-count basis
- Fallback to Close T if next-day open is unavailable (last bar / missing data)

## Execution Costs
- Slippage: 0.05% applied to fill reference price
- Fees: Futu-style US stock estimate (via `OrderCost`)
