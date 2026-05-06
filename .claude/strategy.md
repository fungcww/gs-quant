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

---

## M4.1 Grid Search Findings — Alpha Decay Warning (2026-05-05)

Grid: `sma_window` × `adx_buy_min` (→ `adx_trend_min`) × `vol_risk_fraction` — 36 combinations,
2024 in-sample vs 2025 OOS via `grid_search_optimizer.py`.

### Overfitting signature observed
Top in-sample combinations did **not** transfer to OOS — ranking correlation is strongly negative
among the top tier, which is the hallmark of alpha decay / regime-specific overfitting.

| IS Rank | Params | IS Sharpe | OOS Sharpe | OOS Rank |
|---------|--------|-----------|------------|----------|
| 1 | sma=30, adx=25, vol=0.008 | 1.95 | 0.56 | 12 |
| 2 | sma=30, adx=20, vol=0.008 | 1.91 | 1.07 | 4 |
| 4 | sma=20, adx=20, vol=0.008 | 1.81 | −0.24 | 24 |

### Robust combination (consistent IS → OOS)
`sma=30, adx_trend_min=15` dominated both periods across all vol_risk_fraction levels.
- IS rank 9 (Sharpe 1.69) → OOS rank 1 (Sharpe 1.78) — *improved* out-of-sample.
- IS rank 14 (Sharpe 1.56) → OOS rank 2 (Sharpe 1.74).

### Driver analysis
- **High `adx_trend_min` (25)**: filters to few, high-conviction trend entries that worked in 2024's
  momentum environment but missed 2025's noisier regime — regime-specific overfit.
- **Fast SMA (sma=10)**: high IS ranking collapsed hardest OOS (rank −0.90 Sharpe); too many
  false crossovers in a choppier 2025.
- **High `vol_risk_fraction` (0.008)**: amplifies both IS gains and OOS losses — masks decay in IS,
  magnifies it in OOS. Not an independent signal; interacts with the ADX overfit.

### Rules going forward
1. **Do not select parameters on IS Sharpe rank alone.** Require IS-OOS Sharpe consistency as
   a co-criterion before promoting any combination to live.
2. **Flag any IS Sharpe > 1.80 for extra OOS scrutiny** — in this sweep, every combination
   above that threshold underperformed OOS.
3. **Preferred anchor for further tuning**: `sma=30, adx_trend_min=15`; vol_risk_fraction
   can then be sized to risk-appetite (0.002 = conservative, 0.008 = aggressive).
