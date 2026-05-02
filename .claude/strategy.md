# Strategy Rules (as of M1.9)

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

## Execution Costs
- Slippage: 0.05%
- Fees: Futu-style US stock estimate (via `OrderCost`)
