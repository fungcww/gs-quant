#!/usr/bin/env python3
"""
M4.1 Grid Search Optimizer — sweeps sma_window × adx_buy_min × vol_risk_fraction
for the Stage 32 / M3.2 inverse-vol portfolio and ranks combinations by in-sample (2024) Sharpe.

Parameter note:
  The grid labels `adx_buy_min` to match the prompt, but M3.2 uses regime-switching mode where
  the ADX threshold that controls trend entries is `adx_trend_min` (not `adx_buy_min`, which is
  only consulted in legacy non-regime-switching mode). The grid values are wired to `adx_trend_min`.

Usage:
    python grid_search_optimizer.py                        # in-sample 2024
    python grid_search_optimizer.py --period oos           # out-of-sample 2025
    python grid_search_optimizer.py --db /path/to/market.db --initial-value 20000
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import itertools
import math
from pathlib import Path

import pandas as pd

from gs_quant.backtests.predefined_asset_engine import PredefinedAssetEngine
from gs_quant.backtests.strategy import Strategy

from local_backtest_runner import (
    DEFAULT_SLIPPAGE_FRACTION,
    MACrossoverEODTrigger,
    _TICKERS,
    _annualized_sharpe_from_nav,
    _build_data_manager,
    _nav_total_return_and_max_drawdown,
    _profit_factor_from_backtest,
)

_INITIAL_VALUE = 10_000.0
_IN_SAMPLE_START = dt.date(2024, 1, 1)
_IN_SAMPLE_END = dt.date(2024, 12, 31)
_OOS_START = dt.date(2025, 1, 1)
_OOS_END = dt.date(2025, 12, 31)

PARAM_GRID: dict[str, list] = {
    "sma_window": [10, 20, 30, 50],
    "adx_buy_min": [15, 20, 25],  # wired to adx_trend_min in M3.2 regime-switching mode
    "vol_risk_fraction": [0.002, 0.005, 0.008],
}


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
) -> dict:
    """Run one M3.2 backtest for a single parameter combination and return key metrics.

    Returns a dict with keys: sma_window, adx_buy_min, vol_risk_fraction,
    sharpe, total_return_pct, max_drawdown_pct, profit_factor, num_orders.
    """
    triggers = []
    for symbol, inst in instruments.items():
        hi, lo, _cl, op = hlc_by_symbol[symbol]
        trigger = MACrossoverEODTrigger(
            inst,
            close_by_symbol[symbol],
            sma_window=int(sma_window),
            slippage_fraction=slippage_fraction,
            use_atr_position_sizing=True,
            use_regime_switching=True,
            # adx_buy_min from the grid → adx_trend_min (controls Trend-mode ADX gate in M3.2)
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
    # PredefinedAssetEngine is stateful — create a fresh one per run
    engine = PredefinedAssetEngine(data_mgr=data_manager)

    # Suppress per-trade [M3.2] print lines during batch search
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

    return {
        "sma_window": int(sma_window),
        "adx_buy_min": float(adx_buy_min),
        "vol_risk_fraction": float(vol_risk_fraction),
        "sharpe": sharpe,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd_abs,
        "profit_factor": profit_factor,
        "num_orders": len(backtest.orders),
    }


def run_grid_search(
    db_path: Path,
    *,
    start: dt.date = _IN_SAMPLE_START,
    end: dt.date = _IN_SAMPLE_END,
    initial_value: float = _INITIAL_VALUE,
    param_grid: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Enumerate all combinations in param_grid and return results sorted by Sharpe.

    Data is loaded once; backtests are run sequentially (PredefinedAssetEngine is not picklable).
    """
    if param_grid is None:
        param_grid = PARAM_GRID

    combos = list(itertools.product(
        param_grid["sma_window"],
        param_grid["adx_buy_min"],
        param_grid["vol_risk_fraction"],
    ))
    total = len(combos)

    print(f"=== M4.1 Grid Search — {total} combinations ({start} → {end}) ===")
    print(f"  sma_window:        {param_grid['sma_window']}")
    print(f"  adx_buy_min:       {param_grid['adx_buy_min']}  (→ adx_trend_min in M3.2)")
    print(f"  vol_risk_fraction: {param_grid['vol_risk_fraction']}")
    print(f"  initial_value:     ${initial_value:,.2f}")
    print(f"  tickers ({len(_TICKERS)}):     {', '.join(_TICKERS)}")
    print()

    data_manager, close_by_symbol, instruments, hlc_by_symbol = _build_data_manager(db_path)

    results: list[dict] = []
    for n, (sma_w, adx_min, vol_rf) in enumerate(combos, 1):
        print(
            f"  [{n:>2}/{total}] sma={sma_w:>2}  adx={adx_min:>2}  vol_rf={vol_rf:.3f}",
            end="  ",
            flush=True,
        )
        row = run_single_param_set(
            sma_window=sma_w,
            adx_buy_min=adx_min,
            vol_risk_fraction=vol_rf,
            start=start,
            end=end,
            data_manager=data_manager,
            close_by_symbol=close_by_symbol,
            instruments=instruments,
            hlc_by_symbol=hlc_by_symbol,
            initial_value=initial_value,
        )
        results.append(row)
        sharpe_s = f"{row['sharpe']:.4f}" if math.isfinite(row["sharpe"]) else "N/A"
        ret_s = f"{row['total_return_pct']:+.2f}%" if math.isfinite(row["total_return_pct"]) else "N/A"
        dd_s = f"{row['max_drawdown_pct']:.2f}%" if math.isfinite(row["max_drawdown_pct"]) else "N/A"
        print(f"Sharpe={sharpe_s:>8}  Return={ret_s:>8}  MaxDD={dd_s:>7}")

    df = (
        pd.DataFrame(results)
        .sort_values(["sharpe", "total_return_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4.1 Grid Search Optimizer — Stage 32 / M3.2 parameter sweep"
    )
    parser.add_argument("--db", default="market.db", help="SQLite market database path")
    parser.add_argument(
        "--initial-value",
        type=float,
        default=_INITIAL_VALUE,
        metavar="USD",
        help=f"Initial portfolio equity (default: ${_INITIAL_VALUE:,.0f})",
    )
    parser.add_argument(
        "--period",
        choices=["in-sample", "oos"],
        default="in-sample",
        help="Date range: 'in-sample' = 2024 (default), 'oos' = 2025",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    start, end = (_OOS_START, _OOS_END) if args.period == "oos" else (_IN_SAMPLE_START, _IN_SAMPLE_END)

    results_df = run_grid_search(db_path, start=start, end=end, initial_value=args.initial_value)

    print("\n" + "=" * 80)
    print(f"=== Results — {args.period} ({start} → {end}) ===")
    print("=" * 80)
    with pd.option_context(
        "display.float_format", lambda x: f"{x:.4f}",
        "display.max_columns", None,
        "display.width", 120,
    ):
        print(results_df.to_string(index=True))

    best = results_df.iloc[0]
    print("\n--- Best combination ---")
    print(f"  sma_window:        {int(best['sma_window'])}")
    print(f"  adx_buy_min:       {best['adx_buy_min']:.0f}")
    print(f"  vol_risk_fraction: {best['vol_risk_fraction']:.4f}")
    print(f"  Sharpe:            {best['sharpe']:.4f}")
    print(f"  Total Return:      {best['total_return_pct']:+.2f}%")
    print(f"  Max Drawdown:      {best['max_drawdown_pct']:.2f}%")
    pf_s = f"{best['profit_factor']:.4f}" if math.isfinite(best["profit_factor"]) else "inf"
    print(f"  Profit Factor:     {pf_s}")


if __name__ == "__main__":
    main()
