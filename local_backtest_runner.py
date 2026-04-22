#!/usr/bin/env python3
# customization
"""
Hello-world backtest using PredefinedAssetEngine + SQLiteDataSource (no GsSession / Marquee pricing).

Run from the repository root so paths resolve::

    python local_backtest_runner.py
    python local_backtest_runner.py --quick   # one week, faster smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from gs_quant.backtests.actions import AddTradeAction
from gs_quant.backtests.data_sources import DataManager, SQLiteDataSource
from gs_quant.backtests.core import ValuationFixingType
from gs_quant.backtests.predefined_asset_engine import PredefinedAssetEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.triggers import MktTrigger, MktTriggerRequirements, TriggerDirection
from gs_quant.data.core import DataFrequency
from gs_quant.instrument import EqStock

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


def ensure_market_db(db_path: Path) -> None:
    """Create shared_data/market.db with the canonical schema and sample AAPL rows if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
        # Business days only; flat price above trigger so the demo fires on schedule days.
        for day in pd.bdate_range("2024-01-01", "2024-12-31"):
            d = day.date()
            conn.execute(
                """
                INSERT OR REPLACE INTO market_history (date, symbol, close_price, volume)
                VALUES (?, 'AAPL', ?, ?)
                """,
                (d.strftime("%Y-%m-%d"), 180.0, 1_000_000),
            )
        conn.commit()


def run(start: dt.date, end: dt.date, db_path: Path) -> None:
    ensure_market_db(db_path)

    # Trigger stream: EOD timestamps must match the engine clock for get_data(datetime).
    apple_for_trigger = SQLiteDataSource(
        db_path=str(db_path),
        sql=AAPL_SQL,
        date_column="date",
        value_column="close_price",
        index_at_time=_EOD,
    )
    # Mark-to-market uses calendar dates (DataFrequency.DAILY); execution uses REAL_TIME at EOD.
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

    trigger_req = MktTriggerRequirements(
        data_source=apple_for_trigger,
        trigger_level=150.0,
        direction=TriggerDirection.ABOVE,
    )
    trade_action = AddTradeAction(aapl)
    my_trigger = MktTrigger(trigger_requirements=trigger_req, actions=[trade_action])
    strategy = Strategy(initial_portfolio=None, triggers=[my_trigger])

    engine = PredefinedAssetEngine(data_mgr=data_manager)
    # Large notional: the sample strategy may add size on many session dates when the barrier holds.
    backtest = engine.run_backtest(strategy, start=start, end=end, initial_value=50_000_000.0)

    print(backtest.result_summary.tail(10))
    print("---")
    print("Rows in result_summary:", len(backtest.result_summary))
    print("Orders generated:", len(backtest.orders))
    print(
        "Done (no GsSession). With flat prices, daily Total NAV is unchanged after fills; "
        "change close_price in SQLite to see the level move."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SQLite PredefinedAssetEngine smoke test")
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
    args = parser.parse_args()
    if args.quick:
        start, end = dt.date(2024, 6, 3), dt.date(2024, 6, 7)
    else:
        start, end = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    try:
        run(start, end, args.db.resolve())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
