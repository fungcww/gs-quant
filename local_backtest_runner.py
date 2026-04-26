#!/usr/bin/env python3
# customization
"""
Local backtest using PredefinedAssetEngine + SQLiteDataSource (no GsSession / Marquee pricing).

- Real daily OHLC history for AAPL and SMR via yfinance, persisted to ``market.db``.
- MA crossover vs 20-day SMA: enter on bullish cross (price crosses above SMA); exit when close < SMA.
- Logic gates: buy only when flat; sell full position on exit signal.
- Per-trade fees via OrderCost (Futu-style US stock estimate) so result_summary shows cumulative transaction costs.
- Stage 5 (default full run): 0.05% slippage on fills, 10% hard stop vs entry, pick best SMA on 2024 only, then report 2024 vs 2025 with that window.

Run from the repository root::

    python local_backtest_runner.py              # Stage 5 retail robustness + comparison table
    python local_backtest_runner.py --quick   # one week smoke test
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
from gs_quant.backtests.data_handler import DataHandler
from gs_quant.backtests.order import OrderAtMarket, OrderCost
from gs_quant.backtests.predefined_asset_engine import PredefinedAssetEngine
from gs_quant.backtests.strategy import Strategy
from gs_quant.backtests.triggers import OrdersGeneratorTrigger
from gs_quant.data.core import DataFrequency
from gs_quant.instrument import EqStock
from gs_quant.timeseries.helper import Window
from gs_quant.timeseries.technicals import moving_average

# customization: research-framework utilities (sweep, sharpe, plots)
import math

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# PredefinedAssetEngine uses this wall-clock time for market/valuation events by default.
_EOD = dt.time(23, 0, 0)

_TICKERS = ("AAPL", "SMR")
_DATA_START = "2024-01-01"
_DATA_END_EXCLUSIVE = "2026-01-01"  # yfinance daily ``end`` is exclusive; includes all of 2025-12-31

# customization: retail bid-ask — 0.05% adverse selection vs mid on every market fill
DEFAULT_SLIPPAGE_FRACTION = 0.0005


# customization
class OrderAtMarketWithSlippage(OrderAtMarket):
    """Market order fill at mid ± slippage (buys pay more, sells receive less)."""

    def __init__(
        self,
        instrument,
        quantity: float,
        generation_time: dt.datetime,
        execution_datetime: dt.datetime,
        source: str,
        *,
        slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    ):
        super().__init__(instrument, quantity, generation_time, execution_datetime, source)
        self._slippage_fraction = float(slippage_fraction)

    def _execution_price(self, data_handler: DataHandler) -> float:
        if self.executed_price is None:
            raw = data_handler.get_data(
                self.execution_datetime, self.instrument, ValuationFixingType.PRICE
            )
            q = float(self.quantity)
            slip = self._slippage_fraction
            if q > 1e-12:
                self.executed_price = float(raw * (1.0 + slip))
            elif q < -1e-12:
                self.executed_price = float(raw * (1.0 - slip))
            else:
                self.executed_price = float(raw)
        return self.executed_price


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
    SMA barrier: buy on bullish crossover (close crosses above SMA) when flat;
    sell entire long when close < SMA. Optional hard stop vs entry (retail).
    Brokerage-style fees per routed order leg via OrderCost; fills use ``OrderAtMarketWithSlippage``.
    """

    def __init__(
        self,
        instrument: EqStock,
        close_series: pd.Series,
        *,
        target_usd_allocation: float = 25_000.0,
        sma_window: int = 20,
        fee_model: str = "futu_us_stock",
        eod: dt.time = _EOD,
        # customization: retail robustness (slippage + hard stop vs purchase price)
        slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
        stop_loss_fraction: float = 0.10,
    ):
        super().__init__()
        s = close_series.copy()
        s.index = pd.to_datetime(s.index).normalize()
        self._instrument = instrument
        self._closes = s.sort_index()
        # customization: use library SMA; Window(w, w-1) keeps ramp aligned with strict w-point average (vs int w alone)
        ramp = max(0, sma_window - 1)
        self._sma = moving_average(self._closes, Window(sma_window, ramp)).reindex(self._closes.index)
        # customization: equal-dollar sizing per entry (shares derived from signal-day close)
        self._target_usd = float(target_usd_allocation)
        self._fee_model = str(fee_model)
        self._eod = eod
        self._eps = 1e-9
        self._slippage_fraction = float(slippage_fraction)
        self._stop_loss_fraction = float(stop_loss_fraction)
        # customization: VWAP-style entry reference for stop (mid close × (1 + slip) at buy signal)
        self._purchase_price: float | None = None
        if self._fee_model not in {"futu_us_stock", "none"}:
            raise ValueError(f"Unsupported fee_model={self._fee_model!r}; expected 'futu_us_stock' or 'none'.")

    def get_trigger_times(self) -> list:
        return [self._eod]

    def _execution_fee_usd(self, *, shares: float, price: float, side: str) -> float:
        if self._fee_model == "none":
            return 0.0
        return _futu_us_stock_order_fee_usd(shares=float(shares), price=float(price), side=str(side))

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

        slip = self._slippage_fraction
        buy_px_fee = close_t * (1.0 + slip)
        sell_px_fee = close_t * (1.0 - slip)

        if pos <= self._eps:
            # customization: flat — clear entry reference; only buy on bullish MA cross
            self._purchase_price = None
            if close_prev <= sma_prev and close_t > sma_t:
                if close_t <= self._eps or self._target_usd <= self._eps:
                    return []
                qty = math.floor(self._target_usd / close_t)
                if qty <= 0:
                    return []
                # customization: stop anchor = effective buy (mid + half-spread proxy)
                self._purchase_price = buy_px_fee
                orders.append(
                    OrderAtMarketWithSlippage(
                        self._instrument,
                        float(qty),
                        state,
                        state,
                        "MACrossover",
                        slippage_fraction=slip,
                    )
                )
                fee = self._execution_fee_usd(shares=float(qty), price=buy_px_fee, side="buy")
                if fee > self._eps:
                    orders.append(OrderCost("USD", -abs(fee), "MACrossover", state))
        else:
            # customization: 10% hard stop below purchase (evaluated on mid close)
            stop_hit = (
                self._purchase_price is not None
                and close_t <= self._purchase_price * (1.0 - self._stop_loss_fraction)
            )
            trend_exit = close_t < sma_t
            if stop_hit or trend_exit:
                q = -pos
                if abs(q) > self._eps:
                    self._purchase_price = None
                    orders.append(
                        OrderAtMarketWithSlippage(
                            self._instrument,
                            q,
                            state,
                            state,
                            "MACrossover",
                            slippage_fraction=slip,
                        )
                    )
                    fee = self._execution_fee_usd(shares=float(q), price=sell_px_fee, side="sell")
                    if fee > self._eps:
                        orders.append(OrderCost("USD", -abs(fee), "MACrossover", state))
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


# customization: model 富途牛牛 (Futu) US-stock order fees (approx)
def _futu_us_stock_order_fee_usd(
    *,
    shares: float,
    price: float,
    side: str,
    commission_per_share: float = 0.0049,
    commission_min: float = 0.99,
    platform_per_share: float = 0.005,
    platform_min: float = 1.00,
    settlement_per_share: float = 0.003,
    taf_per_share: float = 0.000166,
    taf_min: float = 0.01,
    taf_max: float = 8.30,
    sec_fee_rate: float = 0.0,
) -> float:
    """
    Estimate 富途牛牛 US-stock fees per filled order leg.

    Assumptions (research approximation):
    - Commission + platform fee + settlement fee apply on both buy/sell.
    - FINRA TAF applies on sells only.
    - SEC fee is set to 0 by default (reported cancelled since 2025-05-14); set `sec_fee_rate` if needed.
    """
    qty = abs(float(shares))
    px = float(price)
    if qty <= 0 or px <= 0:
        return 0.0

    commission = max(commission_min, commission_per_share * qty)
    platform = max(platform_min, platform_per_share * qty)
    settlement = settlement_per_share * qty

    sell_extras = 0.0
    if side.lower() == "sell":
        taf = min(taf_max, max(taf_min, taf_per_share * qty))
        sec = sec_fee_rate * (qty * px) if sec_fee_rate > 0 else 0.0
        sell_extras = taf + sec

    return float(commission + platform + settlement + sell_extras)


def _annualized_sharpe_from_nav(nav: pd.Series, annualization_factor: int = 252) -> float:
    nav = nav.astype(float).dropna()
    if len(nav) < 3:
        return float("nan")
    rets = nav.pct_change().dropna()
    if rets.empty:
        return float("nan")
    vol = float(rets.std())
    if vol <= 0:
        return float("nan")
    return float(math.sqrt(annualization_factor) * float(rets.mean()) / vol)


def _build_data_manager(db_path: Path) -> tuple[DataManager, dict[str, pd.Series], dict[str, EqStock]]:
    ensure_market_db(db_path)
    close_by_symbol = {
        "AAPL": _load_close_series(db_path, AAPL_SQL),
        "SMR": _load_close_series(db_path, SMR_SQL),
    }

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

    instruments = {
        "AAPL": EqStock(name="AAPL", currency="USD", quantity=1),
        "SMR": EqStock(name="SMR", currency="USD", quantity=1),
    }

    data_manager = DataManager()
    data_manager.add_data_source(apple_daily, DataFrequency.DAILY, instruments["AAPL"], ValuationFixingType.PRICE)
    data_manager.add_data_source(apple_rt, DataFrequency.REAL_TIME, instruments["AAPL"], ValuationFixingType.PRICE)
    data_manager.add_data_source(smr_daily, DataFrequency.DAILY, instruments["SMR"], ValuationFixingType.PRICE)
    data_manager.add_data_source(smr_rt, DataFrequency.REAL_TIME, instruments["SMR"], ValuationFixingType.PRICE)

    return data_manager, close_by_symbol, instruments


def _run_single_sma_window(
    *,
    start: dt.date,
    end: dt.date,
    data_manager: DataManager,
    close_by_symbol: dict[str, pd.Series],
    instruments: dict[str, EqStock],
    sma_window: int,
    target_usd_allocation: float,
    fee_model: str,
    initial_value: float,
    slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    stop_loss_fraction: float = 0.10,
):
    triggers = [
        MACrossoverEODTrigger(
            instruments[symbol],
            close_by_symbol[symbol],
            target_usd_allocation=target_usd_allocation,
            sma_window=int(sma_window),
            fee_model=str(fee_model),
            slippage_fraction=float(slippage_fraction),
            stop_loss_fraction=float(stop_loss_fraction),
        )
        for symbol in instruments.keys()
    ]
    strategy = Strategy(initial_portfolio=None, triggers=triggers)
    engine = PredefinedAssetEngine(data_mgr=data_manager)
    backtest = engine.run_backtest(strategy, start=start, end=end, initial_value=float(initial_value))
    summary = backtest.result_summary
    nav = summary["Total NAV"] if "Total NAV" in summary else pd.Series(dtype=float)
    sharpe = _annualized_sharpe_from_nav(nav)
    return backtest, summary, nav, sharpe


def run(
    start: dt.date,
    end: dt.date,
    db_path: Path,
    *,
    sma_windows: list[int] | None = None,
    target_usd_allocation: float = 25_000.0,
    fee_model: str = "futu_us_stock",
    initial_value: float = 50_000_000.0,
    slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    stop_loss_fraction: float = 0.10,
) -> None:
    sma_windows = sma_windows or [20]
    data_manager, close_by_symbol, instruments = _build_data_manager(db_path)

    rows: list[dict] = []
    best_window: int | None = None
    best_sharpe: float = float("-inf")
    best_nav: pd.Series | None = None

    for w in sma_windows:
        backtest, summary, nav, sharpe = _run_single_sma_window(
            start=start,
            end=end,
            data_manager=data_manager,
            close_by_symbol=close_by_symbol,
            instruments=instruments,
            sma_window=int(w),
            target_usd_allocation=float(target_usd_allocation),
            fee_model=str(fee_model),
            initial_value=float(initial_value),
            slippage_fraction=float(slippage_fraction),
            stop_loss_fraction=float(stop_loss_fraction),
        )

        tr, mdd = _nav_total_return_and_max_drawdown(nav)
        final_nav = float(nav.dropna().iloc[-1]) if len(nav.dropna()) else float("nan")
        final_tc = (
            float(summary["Cumulative Transaction Costs"].dropna().iloc[-1])
            if "Cumulative Transaction Costs" in summary and len(summary["Cumulative Transaction Costs"].dropna())
            else 0.0
        )
        rows.append(
            {
                "sma_window": int(w),
                "annualized_sharpe": sharpe,
                "total_return_pct": tr,
                "max_drawdown_pct": abs(mdd),
                "final_total_nav": final_nav,
                "final_cum_transaction_costs": final_tc,
                "orders_generated": len(backtest.orders),
            }
        )

        if sharpe == sharpe and sharpe > best_sharpe:  # not NaN
            best_sharpe = sharpe
            best_window = int(w)
            best_nav = nav

        print(f"--- SMA window = {int(w)} ---")
        print(summary.tail(5))
        print(f"Annualized Sharpe (rf=0): {sharpe:.4f}")
        print(f"Total Return % (Total NAV): {tr:.4f}%")
        print(f"Maximum Drawdown % (Total NAV): {abs(mdd):.4f}%")
        print("Orders generated:", len(backtest.orders))
        print()

    ranking = pd.DataFrame(rows).sort_values(["annualized_sharpe", "total_return_pct"], ascending=[False, False])

    if best_window is not None and best_nav is not None and len(best_nav.dropna()):
        nav = best_nav.dropna().astype(float)
        plt.figure(figsize=(10, 5))
        plt.plot(nav.index, nav.values, linewidth=1.5)
        plt.title(f"Equity Curve (Best SMA window = {best_window}, Sharpe = {best_sharpe:.4f})")
        plt.xlabel("Date")
        plt.ylabel("Total NAV")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest_result.png", dpi=150)
        plt.close()

    print("=== SMA Window Ranking (by Annualized Sharpe, rf=0) ===")
    print(ranking.reset_index(drop=True))
    print("Done (no GsSession).")


# customization: Stage 5 — in-sample SMA selection on 2024, OOS 2025, retail defaults
def run_stage5_retail_robustness(
    db_path: Path,
    *,
    sma_sweep_windows: list[int],
    target_usd_allocation: float = 25_000.0,
    fee_model: str = "futu_us_stock",
    initial_value: float | None = None,
    slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    stop_loss_fraction: float = 0.10,
) -> None:
    """
    Find the best SMA period using **only** 2024, then run that window on 2024 and on **fresh** 2025 data.
    Prints a Sharpe / total-return comparison table (two notional years, same rules).
    """
    iv = float(initial_value) if initial_value is not None else max(100_000.0, 2.2 * float(target_usd_allocation))
    start_2024, end_2024 = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    start_2025, end_2025 = dt.date(2025, 1, 1), dt.date(2025, 12, 31)

    data_manager, close_by_symbol, instruments = _build_data_manager(db_path)

    print("=== Stage 5: SMA sweep on 2024 only (in-sample, rf=0 Sharpe) ===\n")
    sweep_rows: list[dict] = []
    for w in sma_sweep_windows:
        backtest, _summary, nav, sharpe = _run_single_sma_window(
            start=start_2024,
            end=end_2024,
            data_manager=data_manager,
            close_by_symbol=close_by_symbol,
            instruments=instruments,
            sma_window=int(w),
            target_usd_allocation=float(target_usd_allocation),
            fee_model=str(fee_model),
            initial_value=iv,
            slippage_fraction=float(slippage_fraction),
            stop_loss_fraction=float(stop_loss_fraction),
        )
        tr, mdd = _nav_total_return_and_max_drawdown(nav)
        sweep_rows.append(
            {
                "sma_window": int(w),
                "annualized_sharpe": sharpe,
                "total_return_pct": tr,
                "max_drawdown_pct": abs(mdd),
                "orders_generated": len(backtest.orders),
            }
        )
        print(f"--- 2024 SMA window = {int(w)} ---")
        print(f"Annualized Sharpe (rf=0): {sharpe:.4f} | Total Return %: {tr:.4f}% | Max DD %: {abs(mdd):.4f}%")

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["annualized_sharpe", "total_return_pct"], ascending=[False, False], na_position="last"
    )
    print("\n=== 2024 sweep ranking ===")
    print(sweep_df.reset_index(drop=True).to_string())
    if sweep_df.empty:
        raise RuntimeError("No 2024 sweep results.")

    best_w = int(sweep_df.iloc[0]["sma_window"])
    best_sharpe_2024_sweep = float(sweep_df.iloc[0]["annualized_sharpe"])
    print(f"\n>>> Best SMA window from 2024 only: {best_w} (Sharpe in sweep: {best_sharpe_2024_sweep:.4f})\n")

    _, _, nav24, sh24 = _run_single_sma_window(
        start=start_2024,
        end=end_2024,
        data_manager=data_manager,
        close_by_symbol=close_by_symbol,
        instruments=instruments,
        sma_window=best_w,
        target_usd_allocation=float(target_usd_allocation),
        fee_model=str(fee_model),
        initial_value=iv,
        slippage_fraction=float(slippage_fraction),
        stop_loss_fraction=float(stop_loss_fraction),
    )
    _, _, nav25, sh25 = _run_single_sma_window(
        start=start_2025,
        end=end_2025,
        data_manager=data_manager,
        close_by_symbol=close_by_symbol,
        instruments=instruments,
        sma_window=best_w,
        target_usd_allocation=float(target_usd_allocation),
        fee_model=str(fee_model),
        initial_value=iv,
        slippage_fraction=float(slippage_fraction),
        stop_loss_fraction=float(stop_loss_fraction),
    )

    tr24, _ = _nav_total_return_and_max_drawdown(nav24)
    tr25, _ = _nav_total_return_and_max_drawdown(nav25)

    comparison = pd.DataFrame(
        [
            {"Year": "2024", "SMA_window": best_w, "Sharpe_Ratio": sh24, "Total_Return_pct": tr24},
            {"Year": "2025", "SMA_window": best_w, "Sharpe_Ratio": sh25, "Total_Return_pct": tr25},
        ]
    )
    print("=== Comparison: same rules & capital, window fixed from 2024 ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        f"\n(Initial NAV USD {iv:,.0f}; per-symbol target notional USD {target_usd_allocation:,.0f}; "
        f"slippage {slippage_fraction * 100:.3f}%; hard stop {stop_loss_fraction * 100:.1f}% below entry.)"
    )

    if len(nav25.dropna()):
        nav_plot = nav25.dropna().astype(float)
        plt.figure(figsize=(10, 5))
        plt.plot(nav_plot.index, nav_plot.values, linewidth=1.5)
        plt.title(f"2025 OOS equity (SMA={best_w}, Sharpe={sh25:.4f})")
        plt.xlabel("Date")
        plt.ylabel("Total NAV")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest_result.png", dpi=150)
        plt.close()

    print("\nDone (Stage 5, no GsSession).")


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
    parser.add_argument(
        "--initial-value",
        type=float,
        default=None,
        help="Starting portfolio NAV in USD (default: max(100k, 2.2× per-symbol target) for Stage 5)",
    )
    args = parser.parse_args()
    if args.quick:
        start, end = dt.date(2024, 6, 3), dt.date(2024, 6, 7)
        try:
            run(
                start,
                end,
                args.db.resolve(),
                sma_windows=[20],
                target_usd_allocation=25_000.0,
                fee_model="futu_us_stock",
                initial_value=max(100_000.0, 55_000.0),
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            raise
        return
    try:
        run_stage5_retail_robustness(
            args.db.resolve(),
            sma_sweep_windows=[5, 10, 15, 20, 30, 50],
            target_usd_allocation=25_000.0,
            fee_model="futu_us_stock",
            initial_value=args.initial_value,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
