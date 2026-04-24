#!/usr/bin/env python3
# customization
"""
Local backtest using PredefinedAssetEngine + SQLiteDataSource (no GsSession / Marquee pricing).

- Synthetic AAPL close path: geometric random walk (trending) for indicator realism.
- MA crossover vs 20-day SMA: enter on bullish cross (price crosses above SMA); exit when close < SMA.
- Logic gates: buy only when flat; sell full position on exit signal.
- Per-trade fees via OrderCost so result_summary shows cumulative transaction costs.

Run from the repository root::

    python local_backtest_runner.py
    python local_backtest_runner.py --quick   # one week window
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gs_quant.backtests.data_sources import DataManager, SQLiteDataSource
from gs_quant.backtests.core import ValuationFixingType
from gs_quant.backtests.order import OrderAtMarket, OrderCost
from gs_quant.backtests.predefined_asset_engine import PredefinedAssetEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.triggers import OrdersGeneratorTrigger
from gs_quant.data.core import DataFrequency
from gs_quant.instrument import EqStock
from gs_quant.timeseries.helper import Window
from gs_quant.timeseries.technicals import moving_average

# PredefinedAssetEngine uses this wall-clock time for market/valuation events by default.
_EOD = dt.time(23, 0, 0)

AAPL_SQL = """
SELECT date, close_price
FROM market_history
WHERE symbol = 'AAPL'
ORDER BY date
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_market_db(db_path: Path, *, seed: int | None = 42) -> None:
    """Create ``market_history`` if needed and (re)write AAPL closes as a geometric random walk for the demo year."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    # customization: geometric random walk with mild upward drift so the series trends
    log_ret = rng.normal(loc=0.0004, scale=0.012, size=len(days))
    log_level = np.log(150.0) + np.cumsum(log_ret)
    closes = np.exp(log_level)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_history (
                date TEXT,
                symbol TEXT,
                close_price REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
            """
        )
        for i, day in enumerate(days):
            d = day.date()
            conn.execute(
                """
                INSERT OR REPLACE INTO market_history (date, symbol, close_price, volume)
                VALUES (?, 'AAPL', ?, ?)
                """,
                (d.strftime("%Y-%m-%d"), float(closes[i]), 1_000_000),
            )
        conn.commit()


class MACrossoverEODTrigger(OrdersGeneratorTrigger):
    """
    20-day SMA barrier: buy on bullish crossover (close crosses above SMA) when flat;
    sell entire long when close < SMA. Optional fixed fee per routed order leg via OrderCost.
    """

    def __init__(
        self,
        instrument: EqStock,
        close_series: pd.Series,
        *,
        trade_quantity: float = 100.0,
        sma_window: int = 20,
        fee_per_leg_usd: float = 25.0,
        eod: dt.time = _EOD,
    ):
        super().__init__()
        s = close_series.copy()
        s.index = pd.to_datetime(s.index).normalize()
        self._instrument = instrument
        self._closes = s.sort_index()
        # customization: use library SMA; Window(w, w-1) keeps ramp aligned with strict w-point average (vs int w alone)
        ramp = max(0, sma_window - 1)
        self._sma = moving_average(self._closes, Window(sma_window, ramp)).reindex(self._closes.index)
        self._trade_qty = trade_quantity
        self._fee = fee_per_leg_usd
        self._eod = eod
        self._eps = 1e-9

    def get_trigger_times(self) -> list:
        return [self._eod]

    def generate_orders(self, state: dt.datetime, backtest=None):
        d = pd.Timestamp(state.date())
        if d not in self._closes.index:
            return []

        i = int(self._closes.index.get_loc(d))
        if i < 1:
            return []

        close_t = float(self._closes.iloc[i])
        sma_t = self._sma.loc[d]
        if pd.isna(sma_t):
            return []
        sma_t = float(sma_t)
        close_prev = float(self._closes.iloc[i - 1])
        sma_prev = float(self._sma.iloc[i - 1])
        if pd.isna(sma_prev):
            return []

        pos = backtest.holdings.get(self._instrument, 0.0) if backtest is not None else 0.0
        orders: list = []

        if pos <= self._eps:
            # customization: only buy when flat; classic bullish MA cross
            if close_prev <= sma_prev and close_t > sma_t:
                orders.append(
                    OrderAtMarket(
                        instrument=self._instrument,
                        quantity=self._trade_qty,
                        generation_time=state,
                        execution_datetime=state,
                        source='MACrossover',
                    )
                )
                orders.append(OrderCost('USD', -abs(self._fee), 'MACrossover', state))
        else:
            if close_t < sma_t:
                q = -pos
                if abs(q) > self._eps:
                    orders.append(
                        OrderAtMarket(
                            instrument=self._instrument,
                            quantity=q,
                            generation_time=state,
                            execution_datetime=state,
                            source='MACrossover',
                        )
                    )
                    orders.append(OrderCost('USD', -abs(self._fee), 'MACrossover', state))
        return orders


def _load_aapl_close_series(db_path: Path) -> pd.Series:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(AAPL_SQL, conn)
    idx = pd.to_datetime(df["date"]).dt.normalize()
    return pd.Series(df["close_price"].to_numpy(), index=idx, name="close_price").sort_index()


def run(start: dt.date, end: dt.date, db_path: Path, *, seed: int | None) -> None:
    ensure_market_db(db_path, seed=seed)
    close_series = _load_aapl_close_series(db_path)

    apple_daily = SQLiteDataSource(
        db_path=str(db_path),
        sql=AAPL_SQL,
        date_column="date",
        value_column="close_price",
    )
    apple_rt = SQLiteDataSource(
        db_path=str(db_path),
        sql=AAPL_SQL,
        date_column="date",
        value_column="close_price",
        index_at_time=_EOD,
    )

    aapl = EqStock(name="AAPL", currency="USD", quantity=100)

    data_manager = DataManager()
    data_manager.add_data_source(apple_daily, DataFrequency.DAILY, aapl, ValuationFixingType.PRICE)
    data_manager.add_data_source(apple_rt, DataFrequency.REAL_TIME, aapl, ValuationFixingType.PRICE)

    mac_trigger = MACrossoverEODTrigger(
        aapl,
        close_series,
        trade_quantity=float(aapl.quantity),
        sma_window=20,
        fee_per_leg_usd=25.0,
    )
    strategy = Strategy(initial_portfolio=None, triggers=[mac_trigger])

    engine = PredefinedAssetEngine(data_mgr=data_manager)
    backtest = engine.run_backtest(strategy, start=start, end=end, initial_value=50_000_000.0)

    summary = backtest.result_summary
    print(summary.tail(12))
    print("---")
    print("Rows in result_summary:", len(summary))
    if len(summary):
        print("Final Total NAV:", float(summary["Total NAV"].iloc[-1]))
        print("Final Cumulative Transaction Costs:", float(summary["Cumulative Transaction Costs"].iloc[-1]))
    print("Orders generated:", len(backtest.orders))
    print("Done (no GsSession).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SQLite PredefinedAssetEngine MA crossover demo")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a one-week window instead of full-year 2024",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_repo_root() / "shared_data" / "market.db",
        help="Path to SQLite file (default: ./shared_data/market.db)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the random-walk prices (use a different integer to resample paths)",
    )
    args = parser.parse_args()
    if args.quick:
        start, end = dt.date(2024, 6, 3), dt.date(2024, 6, 7)
    else:
        start, end = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    try:
        run(start, end, args.db.resolve(), seed=args.seed)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
