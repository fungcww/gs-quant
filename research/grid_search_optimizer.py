#!/usr/bin/env python3
"""
M4.1 Grid Search Optimizer — parallel parameter sweep for Stage 32 / M3.2.

Parallelism design:
  ProcessPoolExecutor runs one backtest per CPU core concurrently. Because
  PredefinedAssetEngine is stateful and not picklable, it is never sent between
  processes. Instead, each worker process builds its own DataManager once at
  startup via the process initializer (_worker_init), then reuses that local
  state for every task assigned to it. Only primitive task dicts and result dicts
  cross process boundaries.

Market data:
  ensure_market_db() is called once in the main process before the pool starts,
  so workers read from an already-populated market.db (no concurrent yfinance
  downloads). With N workers, the DB is hit exactly N times for schema reads at
  init — not once per combination.

Parameter note:
  The grid uses `adx_buy_min` to match the roadmap label, but M3.2 regime-
  switching mode consults `adx_trend_min` for Trend-mode entries. The two names
  are intentionally synonymous here; `adx_buy_min` values are wired to
  `adx_trend_min` inside run_single_param_set.

Alpha-decay warning (2026-05-05):
  IS top performers (adx=25, high vol_rf) did not transfer to OOS. Use
  analyze_and_report() Stability Peak as a co-criterion before promoting any
  combination. Full analysis in .claude/strategy.md §M4.1 Grid Search Findings.

Usage:
    python grid_search_optimizer.py                   # in-sample 2024, all cores
    python grid_search_optimizer.py --period oos      # out-of-sample 2025
    python grid_search_optimizer.py --workers 4       # limit parallelism
    pip install tqdm                                  # for the live progress bar
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from gs_quant.backtests.core import ValuationFixingType
from gs_quant.backtests.data_sources import DataManager, MissingDataStrategy, SQLiteDataSource
from gs_quant.backtests.predefined_asset_engine import PredefinedAssetEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.data.core import DataFrequency
from gs_quant.instrument import EqStock

from local_backtest_runner import (
    DEFAULT_SLIPPAGE_FRACTION,
    MACrossoverEODTrigger,
    _TICKERS,
    _EOD,
    _annualized_sharpe_from_nav,
    _close_sql,
    _load_close_series,
    _load_symbol_ohlc,
    _nav_total_return_and_max_drawdown,
    _profit_factor_from_backtest,
    _repo_root,
    ensure_market_db,
)

_INITIAL_VALUE = 10_000.0
_IN_SAMPLE_START = dt.date(2024, 1, 1)
_IN_SAMPLE_END = dt.date(2024, 12, 31)
_OOS_START = dt.date(2025, 1, 1)
_OOS_END = dt.date(2025, 12, 31)


def compute_bnh_benchmark(
    close_by_symbol: dict[str, pd.Series],
    start: dt.date,
    end: dt.date,
    initial_value: float = _INITIAL_VALUE,
) -> dict:
    """Equal-weighted buy-and-hold benchmark: buy at day-1 close, hold to end.

    Returns total_return_pct, max_drawdown_pct, sharpe, and the NAV Series.
    $0 commission assumed (single set of trades).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    normalized = []
    for series in close_by_symbol.values():
        s = series.sort_index()
        s = s[(s.index >= start_ts) & (s.index <= end_ts)].astype(float)
        if len(s) < 2 or float(s.iloc[0]) == 0:
            continue
        normalized.append(s / s.iloc[0])
    if not normalized:
        return {
            "total_return_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
            "sharpe": float("nan"),
            "nav": pd.Series(dtype=float),
        }
    combined = pd.concat(normalized, axis=1).ffill()
    nav = combined.mean(axis=1) * initial_value
    sharpe = _annualized_sharpe_from_nav(nav)
    total_return, max_dd = _nav_total_return_and_max_drawdown(nav)
    max_dd_abs = abs(float(max_dd)) if math.isfinite(float(max_dd)) else float("nan")
    return {
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd_abs,
        "sharpe": sharpe,
        "nav": nav,
    }


def plot_equity_curves(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    *,
    title: str = "Strategy vs B&H Benchmark",
    output_path: str | None = None,
) -> None:
    """Overlay strategy equity curve against equal-weight B&H benchmark.

    Green shading = strategy outperforming; red shading = strategy lagging.
    Requires matplotlib; prints a warning and skips silently if not installed.
    """
    if output_path is None:
        output_path = str(_repo_root() / "charts" / "equity_curve_vs_bnh.png")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping equity curve plot (pip install matplotlib)")
        return

    import numpy as np

    # Normalise both indices to midnight — the gs-quant engine stamps NAV at 23:00
    # (EOD trigger time) while the benchmark series uses midnight close dates;
    # without this the intersection is empty and the chart stays blank.
    strat_norm = strategy_nav.copy()
    strat_norm.index = pd.to_datetime(strat_norm.index).normalize()
    bench_norm = benchmark_nav.copy()
    bench_norm.index = pd.to_datetime(bench_norm.index).normalize()

    common = strat_norm.index.intersection(bench_norm.index).sort_values()
    if common.empty:
        print(
            f"WARNING: strategy NAV ({len(strategy_nav)} pts, idx sample: {strategy_nav.index[:3].tolist()}) "
            f"and benchmark NAV ({len(benchmark_nav)} pts) share no common dates — skipping plot."
        )
        return

    strat_vals = np.asarray(strat_norm.reindex(common).values, dtype=np.float64)
    bench_vals = np.asarray(bench_norm.reindex(common).values, dtype=np.float64)
    # Use plain integer positions as x so fill_between never sees dates —
    # matplotlib's unit-conversion path calls np.isfinite on the x array,
    # which raises TypeError for any datetime type regardless of how it entered.
    x_pos = np.arange(len(common), dtype=np.float64)
    dti = pd.DatetimeIndex(common)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_pos, strat_vals, label="Active Strategy (M3.2)", color="steelblue", linewidth=1.5)
    ax.plot(x_pos, bench_vals, label="B&H Equal-Weight", color="darkorange", linewidth=1.5, linestyle="--")
    ax.fill_between(x_pos, strat_vals, bench_vals, where=(strat_vals >= bench_vals), alpha=0.15, color="green", label="Outperforming")
    ax.fill_between(x_pos, strat_vals, bench_vals, where=(strat_vals < bench_vals), alpha=0.15, color="red", label="Underperforming / sitting out")
    # Decorate x-axis with monthly date labels
    tick_mask = [i for i, d in enumerate(dti) if d.day == 1]
    if not tick_mask and len(dti):
        tick_mask = list(range(0, len(dti), max(1, len(dti) // 12)))
    ax.set_xticks(tick_mask)
    ax.set_xticklabels([dti[i].strftime("%Y-%m") for i in tick_mask], rotation=45, ha="right")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Equity curve chart saved → {output_path}")

PARAM_GRID: dict[str, list] = {
    "sma_window": [10, 20, 30, 50],
    "adx_buy_min": [15, 20, 25],  # → adx_trend_min in M3.2 regime-switching mode
    "vol_risk_fraction": [0.002, 0.005, 0.008],
}

# Per-process state — populated once per worker by _worker_init, never pickled
_proc_state: dict | None = None


# ---------------------------------------------------------------------------
# Worker helpers (must be top-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _build_worker_data(db_path: Path) -> dict:
    """Build DataManager + data series from a pre-populated market.db.

    Mirrors the internals of _build_data_manager() but skips ensure_market_db()
    so each worker reads the already-populated DB without re-downloading.
    Concurrent SQLite reads from multiple workers are safe.
    """
    symbols = list(_TICKERS)
    close_by_symbol = {sym: _load_close_series(db_path, _close_sql(sym)) for sym in symbols}
    hlc_by_symbol = {sym: _load_symbol_ohlc(db_path, sym) for sym in symbols}

    _mds = MissingDataStrategy.fill_forward
    instruments: dict[str, EqStock] = {}
    data_manager = DataManager()

    for sym in symbols:
        sql = _close_sql(sym)
        inst = EqStock(name=sym, currency="USD", quantity=1)
        instruments[sym] = inst
        daily_src = SQLiteDataSource(
            db_path=str(db_path),
            sql=sql,
            date_column="date",
            value_column="close_price",
            missing_data_strategy=_mds,
        )
        rt_src = SQLiteDataSource(
            db_path=str(db_path),
            sql=sql,
            date_column="date",
            value_column="close_price",
            index_at_time=_EOD,
            missing_data_strategy=_mds,
        )
        data_manager.add_data_source(daily_src, DataFrequency.DAILY, inst, ValuationFixingType.PRICE)
        data_manager.add_data_source(rt_src, DataFrequency.REAL_TIME, inst, ValuationFixingType.PRICE)

    return {
        "data_manager": data_manager,
        "close_by_symbol": close_by_symbol,
        "instruments": instruments,
        "hlc_by_symbol": hlc_by_symbol,
    }


def _worker_init(db_path_str: str) -> None:
    """ProcessPoolExecutor initializer — runs once per worker process at startup."""
    global _proc_state
    _proc_state = _build_worker_data(Path(db_path_str))


def _worker_run(task: dict) -> dict:
    """Top-level worker task — invoked once per parameter combination.

    Returns a result dict. On exception, returns a NaN row so one failed combo
    does not abort the whole sweep.
    """
    try:
        return run_single_param_set(
            sma_window=task["sma_window"],
            adx_buy_min=task["adx_buy_min"],
            vol_risk_fraction=task["vol_risk_fraction"],
            start=task["start"],
            end=task["end"],
            initial_value=task["initial_value"],
            data_manager=_proc_state["data_manager"],
            close_by_symbol=_proc_state["close_by_symbol"],
            instruments=_proc_state["instruments"],
            hlc_by_symbol=_proc_state["hlc_by_symbol"],
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "sma_window": task["sma_window"],
            "adx_buy_min": task["adx_buy_min"],
            "vol_risk_fraction": task["vol_risk_fraction"],
            "sharpe": float("nan"),
            "total_return_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
            "profit_factor": float("nan"),
            "num_orders": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_param_set(
    *,
    sma_window: int,
    adx_buy_min: float,
    vol_risk_fraction: float,
    start: dt.date,
    end: dt.date,
    data_manager,
    close_by_symbol: dict[str, pd.Series],
    instruments: dict,
    hlc_by_symbol: dict,
    corr_threshold: float = 0.45,
    corr_window: int = 30,
    initial_value: float = _INITIAL_VALUE,
    slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    fee_model: str = "futu_us_stock",
    return_nav: bool = False,
) -> dict:
    """Run one M3.2 backtest for a single parameter combination; return key metrics.

    adx_buy_min is wired to adx_trend_min — the ADX gate for Trend-mode entries
    in M3.2 regime-switching. PredefinedAssetEngine is created fresh here so it
    is never sent across a process boundary.
    """
    triggers = []
    for symbol, inst in instruments.items():
        hi, lo, _cl, op = hlc_by_symbol[symbol]
        trigger = MACrossoverEODTrigger(
            inst,
            close_by_symbol[symbol],
            sma_window=int(sma_window),
            slippage_fraction=slippage_fraction,
            fee_model=fee_model,
            use_atr_position_sizing=True,
            use_regime_switching=True,
            adx_trend_min=float(adx_buy_min),
            high_series=hi,
            low_series=lo,
            open_series=op,
            use_next_day_open=True,
            corr_data=close_by_symbol,
            corr_threshold=float(corr_threshold),
            corr_window=int(corr_window),
            use_vol_sizing=True,
            vol_risk_fraction=float(vol_risk_fraction),
        )
        triggers.append(trigger)

    strategy = Strategy(initial_portfolio=None, triggers=triggers)
    engine = PredefinedAssetEngine(data_mgr=data_manager)

    with contextlib.redirect_stdout(io.StringIO()):
        backtest = engine.run_backtest(
            strategy, start=start, end=end, initial_value=float(initial_value)
        )

    summary = backtest.result_summary
    nav = summary.get("Total NAV", pd.Series(dtype=float))
    sharpe = _annualized_sharpe_from_nav(nav)
    total_return, max_dd = _nav_total_return_and_max_drawdown(nav)
    profit_factor = _profit_factor_from_backtest(backtest)
    max_dd_abs = abs(float(max_dd)) if math.isfinite(float(max_dd)) else float("nan")

    result = {
        "sma_window": int(sma_window),
        "adx_buy_min": float(adx_buy_min),
        "vol_risk_fraction": float(vol_risk_fraction),
        "sharpe": sharpe,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd_abs,
        "profit_factor": profit_factor,
        "num_orders": len(backtest.orders),
    }
    if return_nav:
        result["nav"] = nav
    return result


def run_grid_search(
    db_path: Path,
    *,
    start: dt.date = _IN_SAMPLE_START,
    end: dt.date = _IN_SAMPLE_END,
    initial_value: float = _INITIAL_VALUE,
    param_grid: dict[str, list] | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Run all parameter combinations in parallel; return results sorted by Sharpe.

    Each worker process loads market data from db_path exactly once (via the
    process initializer), not once per task. With W workers and C combinations,
    the DB is read W times total regardless of C.
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    combos = list(itertools.product(
        param_grid["sma_window"],
        param_grid["adx_buy_min"],
        param_grid["vol_risk_fraction"],
    ))
    total = len(combos)
    n_workers = min(max_workers or os.cpu_count() or 1, total)

    print(f"=== M4.1 Grid Search — {total} combinations ({start} → {end}) ===")
    print(f"  sma_window:        {param_grid['sma_window']}")
    print(f"  adx_buy_min:       {param_grid['adx_buy_min']}  (→ adx_trend_min in M3.2)")
    print(f"  vol_risk_fraction: {param_grid['vol_risk_fraction']}")
    print(f"  workers:           {n_workers} / {os.cpu_count()} CPU cores")
    print(f"  initial_value:     ${initial_value:,.2f}")
    print()

    # Populate DB once in the main process — workers read-only from here on
    ensure_market_db(db_path)

    tasks = [
        {
            "sma_window": sma_w,
            "adx_buy_min": adx_min,
            "vol_risk_fraction": vol_rf,
            "start": start,
            "end": end,
            "initial_value": initial_value,
        }
        for sma_w, adx_min, vol_rf in combos
    ]

    results: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(str(db_path),),
    ) as executor:
        future_to_task = {executor.submit(_worker_run, t): t for t in tasks}

        with tqdm(total=total, desc="Grid search", unit="combo") as pbar:
            for future in as_completed(future_to_task):
                row = future.result()
                results.append(row)

                sharpe_s = f"{row['sharpe']:.4f}" if math.isfinite(row.get("sharpe", float("nan"))) else "N/A"
                ret_s = f"{row['total_return_pct']:+.2f}%" if math.isfinite(row.get("total_return_pct", float("nan"))) else "N/A"
                dd_s = f"{row['max_drawdown_pct']:.2f}%" if math.isfinite(row.get("max_drawdown_pct", float("nan"))) else "N/A"
                err = f"  ERROR: {row['error']}" if "error" in row else ""
                tqdm.write(
                    f"  sma={row['sma_window']:>2}  adx={row['adx_buy_min']:>2}"
                    f"  vol_rf={row['vol_risk_fraction']:.3f}"
                    f"  → Sharpe={sharpe_s:>8}  Return={ret_s:>8}  MaxDD={dd_s}{err}"
                )
                pbar.update(1)

    df = (
        pd.DataFrame(results)
        .sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return df


def analyze_and_report(
    df: pd.DataFrame,
    *,
    max_dd_threshold: float = 15.0,
    top_n: int = 10,
    output_csv: str = "grid_results_v1.csv",
    benchmark: dict | None = None,
) -> pd.DataFrame:
    """Filter by max drawdown, identify the Stability Peak, export CSV, print summary.

    Stability Peak: the parameter value that appears most frequently across the
    top_n results after the drawdown filter. A parameter that dominates the top
    tier across many combinations is robust; one that appears only once is lucky.

    When benchmark is supplied (from compute_bnh_benchmark), a B&H row is appended
    to the summary table and a head-to-head comparison is printed.

    Returns the filtered, sorted DataFrame.
    """
    # Export full unfiltered results first
    df.to_csv(output_csv, index=False)

    # Filter and re-sort
    mask = df["max_drawdown_pct"].apply(lambda x: math.isfinite(x) and x <= max_dd_threshold)
    filtered = df[mask].sort_values(
        ["sharpe", "total_return_pct"], ascending=[False, False]
    ).reset_index(drop=True)

    top_df = filtered.head(top_n)

    # Stability Peak: mode + per-value counts for each parameter
    param_cols = ["sma_window", "adx_buy_min", "vol_risk_fraction"]
    peak: dict[str, object] = {}
    counts: dict[str, dict] = {}
    for col in param_cols:
        vc = top_df[col].value_counts().sort_index()
        peak[col] = top_df[col].mode().iloc[0]
        counts[col] = vc.to_dict()

    # ── Filtered table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"=== Filtered Results (MaxDD ≤ {max_dd_threshold}%) — sorted by Sharpe ===")
    print("=" * 80)
    with pd.option_context(
        "display.float_format", lambda x: f"{x:.4f}",
        "display.max_columns", None,
        "display.width", 120,
    ):
        print(filtered.to_string(index=True))

    # ── B&H Benchmark row ─────────────────────────────────────────────────────
    if benchmark:
        bnh_sharpe = benchmark.get("sharpe", float("nan"))
        bnh_ret = benchmark.get("total_return_pct", float("nan"))
        bnh_dd = benchmark.get("max_drawdown_pct", float("nan"))
        sharpe_s = f"{bnh_sharpe:.4f}" if math.isfinite(bnh_sharpe) else "   N/A"
        ret_s = f"{bnh_ret:+.4f}" if math.isfinite(bnh_ret) else "   N/A"
        dd_s = f"{bnh_dd:.4f}" if math.isfinite(bnh_dd) else "   N/A"
        print(
            f"\n  B&H Equal-Weight Benchmark →"
            f"  Sharpe={sharpe_s}  Return={ret_s}%  MaxDD={dd_s}%"
            f"  (commission=$0, all {len(benchmark.get('nav', []))} trading days)"
        )

    # ── Top N ─────────────────────────────────────────────────────────────────
    print(f"\n--- Top {top_n} after drawdown filter ---")
    with pd.option_context(
        "display.float_format", lambda x: f"{x:.4f}",
        "display.max_columns", None,
        "display.width", 120,
    ):
        print(top_df.to_string(index=True))

    # ── Stability Peak ────────────────────────────────────────────────────────
    print(f"\n--- Stability Peak (most frequent params across top {top_n} after DD filter) ---")
    for col in param_cols:
        sorted_counts = sorted(counts[col].items(), key=lambda kv: -kv[1])
        counts_str = "  ".join(f"{k}×{v}" for k, v in sorted_counts)
        print(f"  {col:20s}: {peak[col]}   (top-{top_n} counts: {counts_str})")

    # ── Best Strategy vs Benchmark ────────────────────────────────────────────
    if benchmark and len(filtered) > 0:
        best = filtered.iloc[0]
        bnh_sharpe = benchmark.get("sharpe", float("nan"))
        bnh_ret = benchmark.get("total_return_pct", float("nan"))
        bnh_dd = benchmark.get("max_drawdown_pct", float("nan"))

        def _fmt_sharpe(v: float) -> str:
            return f"{v:.4f}" if math.isfinite(v) else "N/A"

        def _fmt_ret(v: float) -> str:
            return f"{v:+.2f}%" if math.isfinite(v) else "N/A"

        def _fmt_dd(v: float) -> str:
            return f"{v:.2f}%" if math.isfinite(v) else "N/A"

        strat_sharpe = best["sharpe"]
        strat_ret = best["total_return_pct"]
        strat_dd = best["max_drawdown_pct"]

        sharpe_edge = strat_sharpe - bnh_sharpe if (math.isfinite(strat_sharpe) and math.isfinite(bnh_sharpe)) else float("nan")
        ret_edge = strat_ret - bnh_ret if (math.isfinite(strat_ret) and math.isfinite(bnh_ret)) else float("nan")

        print("\n" + "=" * 80)
        print("=== Best Strategy vs B&H Benchmark ===")
        print("=" * 80)
        print(f"  {'':30s}  {'Sharpe':>10}  {'Total Ret':>12}  {'Max DD':>10}")
        print(f"  {'Best Strategy (post-DD filter)':30s}  {_fmt_sharpe(strat_sharpe):>10}  {_fmt_ret(strat_ret):>12}  {_fmt_dd(strat_dd):>10}")
        print(f"  {'B&H Equal-Weight Benchmark':30s}  {_fmt_sharpe(bnh_sharpe):>10}  {_fmt_ret(bnh_ret):>12}  {_fmt_dd(bnh_dd):>10}")
        edge_sharpe_s = f"{sharpe_edge:+.4f}" if math.isfinite(sharpe_edge) else "N/A"
        edge_ret_s = f"{ret_edge:+.2f}%" if math.isfinite(ret_edge) else "N/A"
        print(f"  {'Alpha (Strategy − Benchmark)':30s}  {edge_sharpe_s:>10}  {edge_ret_s:>12}")

    print(f"\nFull results ({len(df)} rows, including filtered-out) → {output_csv}")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4.1 Grid Search — Stage 32 / M3.2 parallel parameter sweep"
    )
    parser.add_argument("--db", default=str(_repo_root() / "shared_data" / "market.db"), help="SQLite market database path")
    parser.add_argument(
        "--initial-value", type=float, default=_INITIAL_VALUE, metavar="USD",
        help=f"Initial portfolio equity (default ${_INITIAL_VALUE:,.0f})",
    )
    parser.add_argument(
        "--period", choices=["in-sample", "oos"], default="in-sample",
        help="'in-sample' = 2024 (default), 'oos' = 2025",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes (default: all CPU cores)",
    )
    parser.add_argument(
        "--max-dd", type=float, default=15.0,
        help="Max drawdown %% filter for the report (default 15)",
    )
    parser.add_argument("--output-csv", default="grid_results_v1.csv")
    args = parser.parse_args()

    db_path = Path(args.db)
    start, end = (_OOS_START, _OOS_END) if args.period == "oos" else (_IN_SAMPLE_START, _IN_SAMPLE_END)

    results_df = run_grid_search(
        db_path,
        start=start,
        end=end,
        initial_value=args.initial_value,
        max_workers=args.workers,
    )

    print("\n" + "=" * 80)
    print(f"=== All Results — {args.period} ({start} → {end}) ===")
    print("=" * 80)
    with pd.option_context(
        "display.float_format", lambda x: f"{x:.4f}",
        "display.max_columns", None,
        "display.width", 120,
    ):
        print(results_df.to_string(index=True))

    # Load market data in main process (workers already exited) for benchmark + plot
    print("\nLoading market data for B&H benchmark computation…")
    main_data = _build_worker_data(db_path)
    benchmark = compute_bnh_benchmark(
        main_data["close_by_symbol"], start=start, end=end, initial_value=args.initial_value
    )
    print(
        f"  B&H benchmark: Sharpe={benchmark['sharpe']:.4f}"
        f"  Return={benchmark['total_return_pct']:+.2f}%"
        f"  MaxDD={benchmark['max_drawdown_pct']:.2f}%"
    )

    analyze_and_report(
        results_df,
        max_dd_threshold=args.max_dd,
        output_csv=args.output_csv,
        benchmark=benchmark,
    )

    best = results_df.iloc[0]
    print("\n--- Overall best combination (no DD filter) ---")
    print(f"  sma_window:        {int(best['sma_window'])}")
    print(f"  adx_buy_min:       {best['adx_buy_min']:.0f}")
    print(f"  vol_risk_fraction: {best['vol_risk_fraction']:.4f}")
    print(f"  Sharpe:            {best['sharpe']:.4f}")
    print(f"  Total Return:      {best['total_return_pct']:+.2f}%")
    print(f"  Max Drawdown:      {best['max_drawdown_pct']:.2f}%")
    pf_s = f"{best['profit_factor']:.4f}" if math.isfinite(best["profit_factor"]) else "inf"
    print(f"  Profit Factor:     {pf_s}")

    # Re-run best combo with return_nav=True to get the equity curve for plotting
    print("\nRe-running best combo to capture equity curve…")
    best_with_nav = run_single_param_set(
        sma_window=int(best["sma_window"]),
        adx_buy_min=float(best["adx_buy_min"]),
        vol_risk_fraction=float(best["vol_risk_fraction"]),
        start=start,
        end=end,
        initial_value=args.initial_value,
        data_manager=main_data["data_manager"],
        close_by_symbol=main_data["close_by_symbol"],
        instruments=main_data["instruments"],
        hlc_by_symbol=main_data["hlc_by_symbol"],
        return_nav=True,
    )
    period_label = args.period.upper()
    chart_path = str(_repo_root() / "charts" / f"equity_curve_vs_bnh_{args.period}.png")
    plot_equity_curves(
        best_with_nav["nav"],
        benchmark["nav"],
        title=(
            f"Best Strategy (sma={int(best['sma_window'])}, adx={best['adx_buy_min']:.0f},"
            f" vol_rf={best['vol_risk_fraction']:.3f}) vs B&H — {period_label} ({start} → {end})"
        ),
        output_path=chart_path,
    )


if __name__ == "__main__":
    main()
