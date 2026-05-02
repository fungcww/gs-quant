# Quant Framework Roadmap

## Current Bookmark: Phase 2 → M2.1

---

### Phase 1: Alpha Discovery ✓ COMPLETE
**Exit criteria met:** Positive Profit Factor confirmed on historical data.

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1.0–M1.8 | Basic crossovers & initial regime switching | Done |
| M1.9 | Asymmetric Mean Reversion — RSI(14)<30 + Lower BB touch entry; Upper BB exit. ADX hysteresis sweep (25/30/35). OOS Win/Loss Ratio. | Done |

---

### Phase 2: The Reality Check ← CURRENT
**Exit criteria:** Strategy viable after execution delays and gaps.

| Milestone | Description | Status |
|-----------|-------------|--------|
| **M2.1** | **Next-Day Open Execution — signals on Close T fill at Open T+1. open_price persisted in market.db. M1.9 vs M2.1 comparison table (Sharpe, Total Return, Profit Factor). Litmus: PF >= 0.90.** | **Current** |
| M2.2 | Transaction Cost & Market Impact Modeling | Pending |

---

### Phase 3: Risk & Portfolio Engineering
**Exit criteria:** Stable equity curve across a basket of diverse assets.

| Milestone | Description |
|-----------|-------------|
| M3.1 | Multi-symbol Correlation Analysis |
| M3.2 | Volatility-Adjusted Position Sizing (Full Portfolio) |

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
