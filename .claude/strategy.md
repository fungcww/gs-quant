# Strategy Rules (as of M2.1)

## Universe
- Single ticker: AAPL / SMR
- Market data: daily OHLC from yfinance, persisted in `market.db`
- Data split: 2024 in-sample, 2025 out-of-sample

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
