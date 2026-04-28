#!/usr/bin/env python3
# customization
"""
Local backtest using PredefinedAssetEngine + SQLiteDataSource (no GsSession / Marquee pricing).

- Real daily OHLC history for AAPL and SMR via yfinance, persisted to ``market.db``.
- MA crossover vs 20-day SMA: enter on bullish cross (price crosses above SMA); exit when close < SMA.
- ADX(14) trend filter on SQLite H/L/C: new buys only when ADX > 20.
- Logic gates: buy only when flat; sell full position on exit signal.
- Per-trade fees via OrderCost (Futu-style US stock estimate) so result_summary shows cumulative transaction costs.
- Stage 6 (default full run): 0.05% slippage, Futu fees, 2024 SMA sweep + neighborhood stability, expanded retail metrics,
  ``stability_heatmap.png`` (2024 Sharpe vs SMA), and 2024/2025 comparison with the in-sample best window.

Run from the repository root::

    python local_backtest_runner.py              # Stage 6 retail robustness + comparison table
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

# customization: research-framework utilities (sweep, sharpe, plots, ADX)
import math

import numpy as np

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
            high = bar.get("High")
            low = bar.get("Low")
            if high is None or (isinstance(high, float) and pd.isna(high)):
                high = close
            if low is None or (isinstance(low, float) and pd.isna(low)):
                low = close
            vol = bar.get("Volume", 0)
            if vol is None or (isinstance(vol, float) and pd.isna(vol)):
                v_int = 0
            else:
                v_int = int(vol)
            d = pd.Timestamp(ts).date().strftime("%Y-%m-%d")
            # customization: persist H/L for ADX(14) from SQLite
            rows.append((d, symbol, float(high), float(low), float(close), v_int))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_history (
                date TEXT,
                symbol TEXT,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
            """
        )
        # customization: migrate older DBs that only had close + volume
        cur = conn.execute("PRAGMA table_info(market_history)")
        col_names = {row[1] for row in cur.fetchall()}
        if "high_price" not in col_names:
            conn.execute("ALTER TABLE market_history ADD COLUMN high_price REAL")
        if "low_price" not in col_names:
            conn.execute("ALTER TABLE market_history ADD COLUMN low_price REAL")
        conn.executemany(
            """
            INSERT OR REPLACE INTO market_history (date, symbol, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


class MACrossoverEODTrigger(OrdersGeneratorTrigger):
    """
    SMA barrier: buy on bullish crossover (close crosses above SMA) when flat;
    sell entire long when close < SMA. Optional hard stop vs entry (retail).
    When ``high_series`` and ``low_series`` are supplied (SQLite H/L with close), ADX(14) is computed
    and **new buys** require ADX > ``adx_buy_min`` (default 20) to reduce entries in non-trending regimes.
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
        # customization: ADX(14) from SQLite H/L/C — new longs only when ADX > threshold (trending regime)
        high_series: pd.Series | None = None,
        low_series: pd.Series | None = None,
        adx_period: int = 14,
        adx_buy_min: float = 20.0,
    ):
        super().__init__()
        s = close_series.copy()
        s.index = pd.to_datetime(s.index).normalize()
        self._instrument = instrument
        self._closes = s.sort_index()
        # customization: ADX on aligned H/L/C (same calendar as closes)
        self._adx_buy_min = float(adx_buy_min)
        self._adx_period = int(adx_period)
        if high_series is not None and low_series is not None:
            hi = high_series.copy()
            lo = low_series.copy()
            hi.index = pd.to_datetime(hi.index).normalize()
            lo.index = pd.to_datetime(lo.index).normalize()
            hi = hi.sort_index().reindex(self._closes.index).astype(float)
            lo = lo.sort_index().reindex(self._closes.index).astype(float)
            hi = hi.fillna(self._closes)
            lo = lo.fillna(self._closes)
            self._adx = _adx_from_hlc(hi, lo, self._closes, period=self._adx_period)
        else:
            # customization: omit gate when H/L not supplied (e.g. legacy smoke tests)
            self._adx = None
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
                # customization: trend filter — skip new buys unless ADX indicates trending market
                if self._adx is not None:
                    adx_raw = self._adx.iloc[i] if i < len(self._adx) else float("nan")
                    adx_t = float(adx_raw) if not pd.isna(adx_raw) else float("nan")
                    if not (adx_t == adx_t) or adx_t <= self._adx_buy_min:
                        return []
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


# customization: H/L/C from SQLite for ADX(14); null legacy highs/lows default to close
def _load_symbol_hlc(db_path: Path, symbol: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT date, high_price, low_price, close_price FROM market_history WHERE symbol = ? ORDER BY date",
            conn,
            params=(symbol,),
        )
    idx = pd.to_datetime(df["date"]).dt.normalize()
    close = pd.to_numeric(df["close_price"], errors="coerce")
    high = pd.to_numeric(df["high_price"], errors="coerce").fillna(close)
    low = pd.to_numeric(df["low_price"], errors="coerce").fillna(close)
    hi = pd.Series(high.to_numpy(), index=idx, name="high_price").sort_index()
    lo = pd.Series(low.to_numpy(), index=idx, name="low_price").sort_index()
    cl = pd.Series(close.to_numpy(), index=idx, name="close_price").sort_index()
    return hi, lo, cl


# customization: Wilder-style ADX (period 14) from high/low/close
def _adx_from_hlc(high: pd.Series, low: pd.Series, close: pd.Series, *, period: int = 14) -> pd.Series:
    h = high.astype(float).sort_index()
    lo = low.astype(float).sort_index()
    c = close.astype(float).sort_index()
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    up_move = h.diff()
    down_move = -lo.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=c.index)
    minus_dm_s = pd.Series(minus_dm, index=c.index)
    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    sm_plus = plus_dm_s.ewm(alpha=alpha, adjust=False).mean()
    sm_minus = minus_dm_s.ewm(alpha=alpha, adjust=False).mean()
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100.0 * (sm_plus / atr_safe)
    minus_di = 100.0 * (sm_minus / atr_safe)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.reindex(c.index)


def _max_drawdown_duration_days(nav: pd.Series) -> int:
    """Longest streak of consecutive calendar days strictly below the running peak NAV."""
    nav = nav.astype(float).dropna()
    if len(nav) < 2:
        return 0
    peak = nav.cummax()
    underwater = (nav < peak).to_numpy()
    best = cur = 0
    for u in underwater:
        if u:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _recovery_factor(nav: pd.Series) -> float:
    """Total return (decimal) divided by max drawdown depth (decimal, positive)."""
    nav = nav.astype(float).dropna()
    if len(nav) < 2:
        return float("nan")
    initial = float(nav.iloc[0])
    final = float(nav.iloc[-1])
    if abs(initial) <= 1e-12:
        return float("nan")
    total_ret = final / initial - 1.0
    running_max = nav.cummax()
    depth = float(1.0 - (nav / running_max).min())
    if depth <= 1e-12:
        return float("inf") if total_ret > 0 else float("nan")
    return float(total_ret / depth)


def _profit_factor_from_backtest(backtest) -> float:
    """Gross wins / gross losses from closed round-trip price PnL in ``trade_ledger`` (fees separate)."""
    try:
        ledger = backtest.trade_ledger()
    except Exception:
        return float("nan")
    if ledger is None or ledger.empty or "Trade PnL" not in ledger.columns:
        return float("nan")
    pnl = ledger.loc[ledger["Status"] == "closed", "Trade PnL"].dropna().astype(float)
    if pnl.empty:
        return float("nan")
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def _expanded_retail_metrics_lines(backtest, nav: pd.Series) -> list[str]:
    tr, mdd = _nav_total_return_and_max_drawdown(nav)
    dd_days = _max_drawdown_duration_days(nav)
    pf = _profit_factor_from_backtest(backtest)
    rf = _recovery_factor(nav)
    if not math.isfinite(pf):
        pf_s = "inf (no losing closed legs)" if pf == float("inf") else "nan"
    else:
        pf_s = f"{pf:.4f}"
    if not math.isfinite(rf):
        rf_s = "inf" if rf == float("inf") else "nan"
    else:
        rf_s = f"{rf:.4f}"
    return [
        f"Max drawdown duration (days underwater): {dd_days}",
        f"Profit factor (sum wins / |sum losses|, ledger PnL): {pf_s}",
        f"Recovery factor (total return / max DD depth): {rf_s}",
        f"Total return % (Total NAV): {tr:.4f}%",
        f"Maximum drawdown % (Total NAV): {abs(mdd):.4f}%",
    ]


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


def _build_data_manager(
    db_path: Path,
) -> tuple[DataManager, dict[str, pd.Series], dict[str, EqStock], dict[str, tuple[pd.Series, pd.Series, pd.Series]]]:
    ensure_market_db(db_path)
    close_by_symbol = {
        "AAPL": _load_close_series(db_path, AAPL_SQL),
        "SMR": _load_close_series(db_path, SMR_SQL),
    }
    # customization: aligned H/L/C per symbol for ADX(14) in triggers
    hlc_by_symbol = {sym: _load_symbol_hlc(db_path, sym) for sym in _TICKERS}

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

    return data_manager, close_by_symbol, instruments, hlc_by_symbol


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
    hlc_by_symbol: dict[str, tuple[pd.Series, pd.Series, pd.Series]] | None = None,
):
    triggers = []
    for symbol in instruments.keys():
        hi = lo = None
        if hlc_by_symbol is not None and symbol in hlc_by_symbol:
            hi, lo, _cl = hlc_by_symbol[symbol]
        triggers.append(
            MACrossoverEODTrigger(
                instruments[symbol],
                close_by_symbol[symbol],
                target_usd_allocation=target_usd_allocation,
                sma_window=int(sma_window),
                fee_model=str(fee_model),
                slippage_fraction=float(slippage_fraction),
                stop_loss_fraction=float(stop_loss_fraction),
                high_series=hi,
                low_series=lo,
            )
        )
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
    data_manager, close_by_symbol, instruments, hlc_by_symbol = _build_data_manager(db_path)

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
            hlc_by_symbol=hlc_by_symbol,
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


# customization: Stage 6 — ADX gate, neighborhood stability, expanded retail metrics, stability heatmap
def _save_stability_heatmap(sweep_df: pd.DataFrame, out_path: str = "stability_heatmap.png") -> None:
    """2024: SMA window (X) vs annualized Sharpe (Y), color-mapped for quick regime read."""
    df = sweep_df.sort_values("sma_window")
    if df.empty:
        return
    y_raw = df["annualized_sharpe"].astype(float).to_numpy()
    x_raw = df["sma_window"].astype(int).to_numpy()
    ok = np.isfinite(y_raw)
    if not np.any(ok):
        return
    x, y = x_raw[ok], y_raw[ok]
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(x, y, c=y, cmap="RdYlGn", s=160, edgecolors="0.35", zorder=3, vmin=float(np.min(y)), vmax=float(np.max(y)))
    ax.plot(x, y, color="0.45", lw=1.1, alpha=0.9, zorder=2)
    fig.colorbar(sc, ax=ax, label="Annualized Sharpe (rf=0)")
    ax.set_xlabel("SMA window")
    ax.set_ylabel("Annualized Sharpe (rf=0)")
    ax.set_title("2024 in-sample: Sharpe vs SMA window (stability heatmap)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
    Stage 6: Find the best SMA on **2024 only**, neighborhood stability sweep, ADX(14)>20 buy filter,
    expanded retail metrics, ``stability_heatmap.png``, then same window on 2025 for OOS context.
    """
    iv = float(initial_value) if initial_value is not None else max(100_000.0, 2.2 * float(target_usd_allocation))
    start_2024, end_2024 = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    start_2025, end_2025 = dt.date(2025, 1, 1), dt.date(2025, 12, 31)

    data_manager, close_by_symbol, instruments, hlc_by_symbol = _build_data_manager(db_path)

    print("=== Stage 6: SMA sweep on 2024 (in-sample, rf=0 Sharpe; ADX>20 buy filter) ===\n")
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
            hlc_by_symbol=hlc_by_symbol,
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

    # customization: neighborhood stability (Best_SMA ± 2, clamped to SMA >= 2)
    neigh_windows = [max(2, best_w + d) for d in (-2, -1, 0, 1, 2)]
    neigh_sharpes: list[float] = []
    print("=== Neighborhood stability (2024 Sharpe, Best_SMA − 2 … Best_SMA + 2) ===\n")
    for w in neigh_windows:
        _, _, nav_n, sh_n = _run_single_sma_window(
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
            hlc_by_symbol=hlc_by_symbol,
        )
        sh_f = float(sh_n) if sh_n == sh_n else float("nan")
        neigh_sharpes.append(sh_f)
        tr_n, mdd_n = _nav_total_return_and_max_drawdown(nav_n)
        print(f"  SMA={int(w)}: Sharpe={sh_f:.6f} | Total Return %: {tr_n:.4f}% | Max DD %: {abs(mdd_n):.4f}%")

    stab_arr = np.array(neigh_sharpes, dtype=float)
    stability_score = float(np.nanstd(stab_arr, ddof=0)) if np.any(np.isfinite(stab_arr)) else float("nan")
    print(f"\nStability score (std dev of neighborhood Sharpes): {stability_score:.6f}")
    if stability_score == stability_score and stability_score > 0.3:
        print("WARNING: High Parameter Sensitivity - Strategy may be overfitted.")

    bt24, _, nav24, sh24 = _run_single_sma_window(
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
        hlc_by_symbol=hlc_by_symbol,
    )
    bt25, _, nav25, sh25 = _run_single_sma_window(
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
        hlc_by_symbol=hlc_by_symbol,
    )

    tr24, _ = _nav_total_return_and_max_drawdown(nav24)
    tr25, _ = _nav_total_return_and_max_drawdown(nav25)

    comparison = pd.DataFrame(
        [
            {"Year": "2024", "SMA_window": best_w, "Sharpe_Ratio": sh24, "Total_Return_pct": tr24},
            {"Year": "2025", "SMA_window": best_w, "Sharpe_Ratio": sh25, "Total_Return_pct": tr25},
        ]
    )
    print("\n=== Comparison: same rules & capital, window fixed from 2024 ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        f"\n(Initial NAV USD {iv:,.0f}; per-symbol target notional USD {target_usd_allocation:,.0f}; "
        f"slippage {slippage_fraction * 100:.3f}%; hard stop {stop_loss_fraction * 100:.1f}% below entry; "
        f"ADX(14) buy gate > 20.)"
    )

    print("\n=== Expanded retail metrics (best SMA from 2024) ===")
    print("--- 2024 ---")
    for line in _expanded_retail_metrics_lines(bt24, nav24):
        print(line)
    print("--- 2025 ---")
    for line in _expanded_retail_metrics_lines(bt25, nav25):
        print(line)

    _save_stability_heatmap(sweep_df, "stability_heatmap.png")
    print("\nWrote stability_heatmap.png (2024 Sharpe vs SMA window).")

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

    print("\nDone (Stage 6, no GsSession).")


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
        help="Starting portfolio NAV in USD (default: max(100k, 2.2× per-symbol target) for Stage 6)",
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
