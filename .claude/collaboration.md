# Collaboration Notes

The user co-pilots with a Gemini AI agent — Gemini handles high-level planning and produces prompts/tasks, Claude is the implementation and critique layer.

## How to apply
- Treat pasted Gemini prompts as implementation requests, not gospel — evaluate them critically before acting.
- Freely raise concerns if a Gemini suggestion seems architecturally unsound, risky, redundant, or inconsistent with the existing codebase.
- Debate the decision if there is a clearly better alternative; briefly explain the tradeoff.
- Ask for clarification when the prompt is ambiguous or missing context needed to implement it correctly.
- Do not blindly implement something that could introduce bugs, security issues, or unnecessary complexity just because Gemini suggested it.

## Project Layout Rule
All self-developed quant research code belongs in `/workspaces/gs-quant/research/`, split into two tiers:

```
research/
├── engines/               ← Reusable library modules (importable, stable)
│   ├── __init__.py
│   ├── data_loader.py         (HistoricalDataLoader)
│   ├── beta_engine.py         (BetaCalculator)
│   └── output_manager.py      (OutputManager — enforces outputs/ layout)
└── experiments/           ← Runnable research scripts (not imported by engines)
    ├── __init__.py
    ├── local_backtest_runner.py
    ├── grid_search_optimizer.py
    ├── tournament_optimizer.py
    └── outputs/           ← ALL experiment write operations go here
        ├── scratch/           hot, raw, iterative data (trade logs, temp CSVs, SQLite)
        └── artifacts/         finalized outputs (summary CSVs, equity-curve PNGs)
```

- **engines/**: reusable, importable modules. Other code imports from here.
- **experiments/**: one-off runnable scripts. May import from `engines/` or from each other, but nothing imports from `experiments/` except other experiments.
- Import path: `from research.engines.data_loader import HistoricalDataLoader` — works because `research` is registered as a package in `pyproject.toml` and the repo is installed with `pip install -e .`. Never use `sys.path` hacks.
- Never create strategy, backtest, or analysis scripts at the project root or any other directory.

If a required mathematical formula, indicator, or calculation method is genuinely absent from `gs_quant/`, extend or add it directly inside `/workspaces/gs-quant/gs_quant/` (e.g. `gs_quant/timeseries/` or `gs_quant/analytics/`) rather than reimplementing it in the research script. This keeps the codebase coherent, avoids redundant imports, and means future scripts can reuse the extension via the standard gs-quant import path.

## Output Separation Rule
All write operations from experiment scripts must go through `OutputManager` from `research.engines.output_manager`. Two tiers are enforced:

- **`scratch/`** — ephemeral, local-only: raw trade logs, iterative CSVs, SQLite snapshots. Never promoted to production.
- **`artifacts/`** — finalized: summary metric CSVs, equity-curve PNGs ready for review or DB sync.

```python
from research.engines.output_manager import OutputManager
om = OutputManager()
chart_path   = om.artifact_path(f"tournament_{ts}.png")   # PNG equity curve
summary_path = om.artifact_path("comparison_report.csv")  # final metrics
log_path     = om.scratch_path("trades_raw.csv")          # trade-by-trade log
```

Never write charts or CSVs to the project root, `research/`, or the legacy `charts/` directory. This applies to all scripts — including standalone engine runs such as `beta_engine.py`.

**Production sync stub**: after identifying the tournament champion, call `sync_champion_to_core_db(metrics_dict)`. This stub logs the payload and is wired to push only the winner's aggregated metadata to the Linux PostgreSQL server — raw scratch data never leaves local storage.

## Futu Trading Fee Rule
Every trade simulation must account for Futu Securities trading fees. Never model execution costs with slippage alone.
- **Fee structure**: apply Futu-style US stock commission via the `OrderCost` helper (already wired in the backtest runner).
- When adding new execution paths, new stages, or new backtesting scripts, always pass fees through `OrderCost` — do not default to zero-fee or slippage-only models.
- If a new script does not yet use `OrderCost`, flag it and add the fee before reporting results.

## Historical Data Rule
Never call yfinance, pandas-datareader, or any other third-party data provider directly to fetch historical OHLCV prices. All historical market data must go through `HistoricalDataLoader` in `research/engines/data_loader.py`.

```python
from research.engines.data_loader import HistoricalDataLoader
loader = HistoricalDataLoader()
df = loader.get_ohlcv(ticker, start_date, end_date)
```

`HistoricalDataLoader` hits the internal API server (`API_SERVER_URL` in `.env`) and falls back to `shared_data/market.db` if the server is unreachable. This applies to every script — new and existing. If an existing script calls yfinance directly, migrate it to `HistoricalDataLoader` before extending it.

The one exception is `ensure_hk_data()` in `beta_engine.py`, which is the DB seed function that pre-dates this rule; do not add new yfinance call-sites modelled after it.

## gs-quant Library-First Rule
Before building any new strategy component, indicator, or mathematical calculation:
1. **Search the gs-quant library first** — check `gs_quant/timeseries/`, `gs_quant/analytics/`, `gs_quant/risk/`, and `gs_quant/markets/` for existing implementations.
2. **Use it if it exists** — do not rewrite math (e.g. volatility, correlation, regression, convexity) that gs-quant already provides; import and call the library function instead.
3. **Only build custom code** when the required logic is genuinely absent from gs-quant or requires project-specific adaptation that cannot be cleanly layered on top.
4. When uncertain, do a quick grep/search of the gs-quant source before concluding something is missing.
