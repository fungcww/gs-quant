#!/usr/bin/env python3
# customization
"""
Local backtest using PredefinedAssetEngine + SQLiteDataSource (no GsSession / Marquee pricing).

- Real daily OHLC history for AAPL and SMR via yfinance, persisted to ``market.db``.
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

import pandas as pd

from gs_quant.backtests.data_sources import DataManager, MissingDataStrategy, SQLiteDataSource
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

_TICKERS = ("AAPL", "SMR")
_DATA_START = "2024-01-01"
_DATA_END_EXCLUSIVE = "2026-01-01"  # yfinance daily ``end`` is exclusive; includes all of 2025-12-31

AAPL_SQL = """
SELECT date, close_price
FROM market_history
WHERE symbol = 'AAPL'
ORDER BY date
"""

SMR_SQL = """
SELECT date, close_price
FROM market_history
WHERE symbol = 'SMR'
ORDER BY date
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_market_db(db_path: Path) -> None:
    """Create ``market_history`` if needed and load AAPL/SMR daily closes from yfinance into ``market.db``."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "yfinance is required for real-world data ingestion. Install with: pip install yfinance"
        ) from e

    db_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, float, int]] = []

    for symbol in _TICKERS:
        hist = yf.Ticker(symbol).history(start=_DATA_START, end=_DATA_END_EXCLUSIVE, auto_adjust=False)
        if hist.empty:
            raise RuntimeError(f"yfinance returned no rows for {symbol!r}; check connectivity or ticker.")
        for ts, bar in hist.iterrows():
            close = bar.get("Close")
            if close is None or (isinstance(close, float) and pd.isna(close)):
                continue
            vol = bar.get("Volume", 0)
            if vol is None or (isinstance(vol, float) and pd.isna(vol)):
                v_int = 0
            else:
                v_int = int(vol)
            d = pd.Timestamp(ts).date().strftime("%Y-%m-%d")
            rows.append((d, symbol, float(close), v_int))

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
        conn.executemany(
            """
            INSERT OR REPLACE INTO market_history (date, symbol, close_price, volume)
            VALUES (?, ?, ?, ?)
            """,
            rows,
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
        d = pd.Timestamp(state.date()).normalize()
        idx_arr = self._closes.index.get_indexer([d], method=None)
        i = int(idx_arr[0]) if len(idx_arr) else -1
        if i < 0 or i < 1:
            return []

        try:
            close_t = float(self._closes.iloc[i])
            sma_t_raw = self._sma.iloc[i]
            close_prev = float(self._closes.iloc[i - 1])
            sma_prev_raw = self._sma.iloc[i - 1]
        except (KeyError, IndexError, TypeError):
            return []

        if pd.isna(sma_t_raw) or pd.isna(sma_prev_raw):
            return []
        sma_t = float(sma_t_raw)
        sma_prev = float(sma_prev_raw)

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


def _load_close_series(db_path: Path, sql: str) -> pd.Series:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    idx = pd.to_datetime(df["date"]).dt.normalize()
    return pd.Series(df["close_price"].to_numpy(), index=idx, name="close_price").sort_index()


def _nav_total_return_and_max_drawdown(nav: pd.Series) -> tuple[float, float]:
    """Return (total_return_pct, max_drawdown_pct) for Total NAV; max drawdown is worst peak-to-trough return in %."""
    nav = nav.astype(float).dropna()
    if nav.empty:
        return float("nan"), float("nan")
    initial = float(nav.iloc[0])
    final = float(nav.iloc[-1])
    total_ret = (final / initial - 1.0) * 100.0 if abs(initial) > 1e-12 else float("nan")
    running_max = nav.cummax()
    rel = nav / running_max - 1.0
    max_dd_pct = float(rel.min() * 100.0)
    return total_ret, max_dd_pct


def run(start: dt.date, end: dt.date, db_path: Path) -> None:
    ensure_market_db(db_path)
    close_aapl = _load_close_series(db_path, AAPL_SQL)
    close_smr = _load_close_series(db_path, SMR_SQL)

    # fill_forward: engine marks to market on calendar days (e.g. holidays) and RT series uses EOD clock, not midnight
    _mds = MissingDataStrategy.fill_forward
    apple_daily = SQLiteDataSource(
        db_path=str(db_path),
        sql=AAPL_SQL,
        date_column="date",
        value_column="close_price",
        missing_data_strategy=_mds,
    )
    apple_rt = SQLiteDataSource(
        db_path=str(db_path),
        sql=AAPL_SQL,
        date_column="date",
        value_column="close_price",
        index_at_time=_EOD,
        missing_data_strategy=_mds,
    )
    smr_daily = SQLiteDataSource(
        db_path=str(db_path),
        sql=SMR_SQL,
        date_column="date",
        value_column="close_price",
        missing_data_strategy=_mds,
    )
    smr_rt = SQLiteDataSource(
        db_path=str(db_path),
        sql=SMR_SQL,
        date_column="date",
        value_column="close_price",
        index_at_time=_EOD,
        missing_data_strategy=_mds,
    )

    aapl = EqStock(name="AAPL", currency="USD", quantity=100)
    smr = EqStock(name="SMR", currency="USD", quantity=100)

    data_manager = DataManager()
    data_manager.add_data_source(apple_daily, DataFrequency.DAILY, aapl, ValuationFixingType.PRICE)
    data_manager.add_data_source(apple_rt, DataFrequency.REAL_TIME, aapl, ValuationFixingType.PRICE)
    data_manager.add_data_source(smr_daily, DataFrequency.DAILY, smr, ValuationFixingType.PRICE)
    data_manager.add_data_source(smr_rt, DataFrequency.REAL_TIME, smr, ValuationFixingType.PRICE)

    mac_aapl = MACrossoverEODTrigger(
        aapl,
        close_aapl,
        trade_quantity=float(aapl.quantity),
        sma_window=20,
        fee_per_leg_usd=25.0,
    )
    mac_smr = MACrossoverEODTrigger(
        smr,
        close_smr,
        trade_quantity=float(smr.quantity),
        sma_window=20,
        fee_per_leg_usd=25.0,
    )
    strategy = Strategy(initial_portfolio=None, triggers=[mac_aapl, mac_smr])

    engine = PredefinedAssetEngine(data_mgr=data_manager)
    backtest = engine.run_backtest(strategy, start=start, end=end, initial_value=50_000_000.0)

    summary = backtest.result_summary
    print(summary.tail(12))
    print("---")
    print("Rows in result_summary:", len(summary))
    if len(summary):
        print("Final Total NAV:", float(summary["Total NAV"].iloc[-1]))
        print("Final Cumulative Transaction Costs:", float(summary["Cumulative Transaction Costs"].iloc[-1]))
        tr, mdd = _nav_total_return_and_max_drawdown(summary["Total NAV"])
        print(f"Total Return % (Total NAV): {tr:.4f}%")
        # mdd is the worst (nav/peak - 1)*100, <= 0; report magnitude as conventional "max drawdown %"
        print(f"Maximum Drawdown % (Total NAV): {abs(mdd):.4f}%")
    print("Orders generated:", len(backtest.orders))
    print("Done (no GsSession).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SQLite PredefinedAssetEngine MA crossover demo")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a one-week window instead of the full configured range",
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
        start, end = dt.date(2024, 1, 1), dt.date(2025, 12, 31)
    try:
        run(start, end, args.db.resolve())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
