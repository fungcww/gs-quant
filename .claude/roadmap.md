# Quant Framework Roadmap

## Current Bookmark: Phase 3 → M3.2

---

### Phase 1: Alpha Discovery ✓ COMPLETE
**Exit criteria met:** Positive Profit Factor confirmed on historical data.

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1.0–M1.8 | Basic crossovers & initial regime switching | Done |
| M1.9 | Asymmetric Mean Reversion — RSI(14)<30 + Lower BB touch entry; Upper BB exit. ADX hysteresis sweep (25/30/35). OOS Win/Loss Ratio. | Done |

---

### Phase 2: The Reality Check ✓ COMPLETE
**Exit criteria met:** Strategy viable after execution delays and gaps.

| Milestone | Description | Status |
|-----------|-------------|--------|
| M2.1 | Next-Day Open Execution — signals on Close T fill at Open T+1. open_price persisted in market.db. M1.9 vs M2.1 comparison table (Sharpe, Total Return, Profit Factor). Litmus: PF >= 0.90. | Done |
| M2.2 | Transaction Cost & Market Impact Modeling | Pending |

---

### Phase 3: Risk & Portfolio Engineering ← CURRENT
**Exit criteria:** Stable equity curve across a basket of diverse assets.

| Milestone | Description | Status |
|-----------|-------------|--------|
| M3.1 | 10-ticker universe (AAPL, SMR, NVDA, TSLA, GOOGL, MSFT, META, AMD, NNE, OKLO). Pearson correlation matrix (2024 in-sample). 30-day rolling correlation gate (threshold 0.70). Portfolio Total NAV report + Diversification Benefit metric. Parallel yfinance download. | Done |
| **M3.2** | **Retail Inverse-Vol Sizing. Initial equity $10K default (`--initial-value`). 20-day rolling std sizing: risk_usd = 0.5% NAV ÷ (price × vol). Fractional shares. corr_threshold tightened to 0.45. Position notional printed per trade. ATR Chandelier stop preserved.** | **Current** |

---

### Phase 4: High-Performance Infrastructure
**Exit criteria:** Stress-test 10,000+ parameter combinations in seconds.

| Milestone | Description |
|-----------|-------------|
| M4.1 | Vectorized Backtesting (Universe-scale) |
| M4.2 | Intraday/Tick-level Analysis |

---

## Stage Map (local_backtest_runner.py)

| Stage | Description |
|-------|-------------|
| 6 | Full run — 0.05% slippage, Futu fees, 2024 SMA sweep + neighbourhood stability, retail metrics, `stability_heatmap.png`, 2024/2025 comparison |
| 8 | Regime Switching — ADX gear-shifter: Trend (ADX>25, SMA crossover) vs Mean Reversion (ADX<20, BB lower touch). ATR sizing + Chandelier stop. |
| 9 | Asymmetric Mean Reversion — RSI(14)<30 + BB lower touch entry; Upper BB exit. ADX threshold hysteresis sweep. OOS Win/Loss Ratio. |
| 10 / M2.1 | Next-Day Open Execution — `open_price` added to market.db; fills shift from signal-day Close to T+1 Open via `override_fill_price`. M1.9 vs M2.1 comparison printed at run end. |
| 31 / M3.1 | 10-ticker universe; parallel yfinance download; Pearson correlation matrix (2024); 30-day rolling corr gate (0.70); combined portfolio Total NAV + Diversification Benefit. |
| 32 / M3.2 | **Default stage.** Retail inverse-vol sizing ($10K equity, corr 0.45, fractional shares, 0.5% daily-vol-risk per trade). `--stage 31` restores M3.1. |
