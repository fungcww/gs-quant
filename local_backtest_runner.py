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
- Stage 8 (Regime Switching): ADX(14) gear-shifter — Trend mode (ADX>25, SMA crossover) vs Mean Reversion
  (ADX<20, Bollinger Band lower touch / SMA return exit). ATR sizing + Chandelier stop preserved in both modes.
- Stage 9 (Asymmetric Mean Reversion): Tightens the Mean Reversion entry to require RSI(14)<30 in addition to
  the Lower Bollinger Band touch (exhaustion confirmation). Exits at the Upper Bollinger Band instead of the SMA
  for an asymmetric risk/reward profile. Includes a hysteresis sweep over ADX Trend Threshold (25/30/35) and
  reports the 2025 OOS Win/Loss Ratio to validate the asymmetric payout improvement.
- Stage 10 / M2.1 (Next-Day Open Execution): Removes look-ahead bias by shifting all order fills from
  signal-day Close to the Open of Day T+1. ``open_price`` is persisted in ``market.db`` via yfinance.
  A comparison table (M1.9 Close vs M2.1 Open) is printed for 2025 OOS Sharpe, Total Return, and Profit
  Factor. Litmus test: Profit Factor >= 0.90 confirms the strategy is not scalping unreachable overnight gaps.

Run from the repository root::

    python local_backtest_runner.py              # Stage 9 full run (retail robustness + regime comparison)
    python local_backtest_runner.py --quick   # one week smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# M2.1: next-day open execution time (signals fire at EOD, fills execute at T+1 open)
_MARKET_OPEN = dt.time(9, 30, 0)


_TICKERS = ("AAPL", "SMR", "NVDA", "TSLA", "GOOGL", "MSFT", "META", "AMD", "NNE", "OKLO")
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
        override_fill_price: float | None = None,
    ):
        super().__init__(instrument, quantity, generation_time, execution_datetime, source)
        self._slippage_fraction = float(slippage_fraction)
        # M2.1: pre-computed fill price (slippage already included) bypasses data_handler lookup
        self._override_fill_price = float(override_fill_price) if override_fill_price is not None else None

    def _execution_price(self, data_handler: DataHandler) -> float:
        if self.executed_price is None:
            if self._override_fill_price is not None:
                self.executed_price = self._override_fill_price
            else:
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


def _close_sql(symbol: str) -> str:
    """Parameterised close-price query for a single symbol (symbol is always from _TICKERS)."""
    return f"SELECT date, close_price FROM market_history WHERE symbol = '{symbol}' ORDER BY date"





def _repo_root() -> Path:
    return Path(__file__).resolve().parent


# All generated charts are written here (created on first run if absent)
_CHARTS_DIR = _repo_root() / "charts"


def _fetch_ticker_history(symbol: str, start: str, end: str) -> tuple[str, "pd.DataFrame"]:
    """Download OHLCV for one ticker; runs in a thread (I/O-bound)."""
    import yfinance as yf  # type: ignore[import-untyped]
    hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    return symbol, hist


def ensure_market_db(db_path: Path) -> None:
    """Create ``market_history`` if needed and load all _TICKERS daily OHLC from yfinance into ``market.db``.

    Downloads are issued in parallel via ThreadPoolExecutor (I/O-bound); indicator pre-computation
    (ADX, RSI, SMA) is already vectorised over the full series so no further parallelism is needed there.
    Backtests are kept sequential because PredefinedAssetEngine holds mutable state that is not picklable.
    """
    try:
        import yfinance as yf  # noqa: F401 — validate install before spawning threads
    except ImportError as e:
        raise RuntimeError(
            "yfinance is required for real-world data ingestion. Install with: pip install yfinance"
        ) from e

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Parallel download — one thread per ticker, capped at 8 to avoid rate-limiting
    print(f"Fetching {len(_TICKERS)} tickers from yfinance (parallel)…")
    hist_by_sym: dict[str, "pd.DataFrame"] = {}
    with ThreadPoolExecutor(max_workers=min(len(_TICKERS), 8)) as pool:
        futures = {
            pool.submit(_fetch_ticker_history, sym, _DATA_START, _DATA_END_EXCLUSIVE): sym
            for sym in _TICKERS
        }
        for fut in as_completed(futures):
            sym, hist = fut.result()
            if hist.empty:
                raise RuntimeError(f"yfinance returned no rows for {sym!r}; check connectivity or ticker.")
            hist_by_sym[sym] = hist
            print(f"  {sym}: {len(hist)} bars")

    rows: list[tuple[str, str, float, float, float, float, int]] = []
    for symbol in _TICKERS:
        hist = hist_by_sym[symbol]
        for ts, bar in hist.iterrows():
            close = bar.get("Close")
            if close is None or (isinstance(close, float) and pd.isna(close)):
                continue
            open_px = bar.get("Open")
            if open_px is None or (isinstance(open_px, float) and pd.isna(open_px)):
                open_px = close
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
            # customization: persist O/H/L for M2.1 next-day open execution and ADX(14)
            rows.append((d, symbol, float(open_px), float(high), float(low), float(close), v_int))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_history (
                date TEXT,
                symbol TEXT,
                open_price REAL,
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
        if "open_price" not in col_names:
            conn.execute("ALTER TABLE market_history ADD COLUMN open_price REAL")
        conn.executemany(
            """
            INSERT OR REPLACE INTO market_history (date, symbol, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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
    Stage 9 (``use_regime_switching=True``): ADX gear-shifter — ADX>adx_trend_min uses SMA crossover (Trend
    mode); ADX<adx_mean_rev_max uses Lower Bollinger Band touch **and** RSI(14)<30 as a dual-confirmation
    entry (Asymmetric Mean Reversion mode). Exit for mean-reversion positions is the Upper Bollinger Band
    rather than the SMA, creating an asymmetric risk/reward target.
    ATR sizing and Chandelier trailing stop are preserved in both modes.
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
        # customization: Stage 7 volatility-adjusted risk management (ATR sizing + Chandelier trailing stop)
        use_atr_position_sizing: bool = False,
        atr_period: int = 20,
        # customization: TODO — optional time-varying fraction (vs this fixed scalar). Today it only scales *entry* size
        #   (risk_usd / ATR → smaller qty when ATR is high). A richer policy might lower the fraction after drawdowns,
        #   realized-vol spikes, or a recent stop-out so the next entry accepts less notional “risk budget” until calm.
        atr_risk_fraction_of_nav: float = 0.005,
        atr_max_allocation_fraction_of_nav: float = 0.50,
        chandelier_atr_multiple: float = 3.0,
        # customization: ADX(14) from SQLite H/L/C — new longs only when ADX > threshold (trending regime)
        high_series: pd.Series | None = None,
        low_series: pd.Series | None = None,
        adx_period: int = 14,
        adx_buy_min: float = 20.0,
        # Stage 8/9 regime switching
        use_regime_switching: bool = False,
        adx_trend_min: float = 25.0,
        adx_mean_rev_max: float = 20.0,
        # Stage 9: RSI(14) exhaustion filter for mean-reversion entries
        rsi_period: int = 14,
        # M2.1: execute fills at next-day open instead of signal-day close
        use_next_day_open: bool = False,
        open_series: pd.Series | None = None,
        # M3.1: correlation gate — block new buys when 30-day rolling corr to any held position >= threshold
        corr_data: dict[str, pd.Series] | None = None,
        corr_threshold: float = 0.70,
        corr_window: int = 30,
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
        # Stage 8 regime switching
        self._use_regime_switching = bool(use_regime_switching)
        self._adx_trend_min = float(adx_trend_min)
        self._adx_mean_rev_max = float(adx_mean_rev_max)
        self._entry_mode: str | None = None
        self._lower_band: pd.Series | None = None
        self._upper_band: pd.Series | None = None
        self._rsi: pd.Series | None = None
        if self._use_regime_switching:
            if self._adx is None:
                raise ValueError("Stage 9 regime switching requires high_series and low_series for ADX.")
            _std = self._closes.rolling(int(sma_window), min_periods=int(sma_window)).std()
            self._lower_band = (self._sma - 2.0 * _std).reindex(self._closes.index)
            self._upper_band = (self._sma + 2.0 * _std).reindex(self._closes.index)
            # Stage 9: RSI(14) exhaustion confirmation for mean-reversion entries
            self._rsi = _rsi_from_close(self._closes, period=rsi_period)
        # customization: equal-dollar sizing per entry (shares derived from signal-day close)
        self._target_usd = float(target_usd_allocation)
        self._fee_model = str(fee_model)
        self._eod = eod
        self._eps = 1e-9
        self._slippage_fraction = float(slippage_fraction)
        self._stop_loss_fraction = float(stop_loss_fraction)
        # customization: VWAP-style entry reference for stop (mid close × (1 + slip) at buy signal)
        self._purchase_price: float | None = None
        # customization: Stage 7 state (ATR sizing + Chandelier trailing stop)
        self._use_atr_position_sizing = bool(use_atr_position_sizing)
        self._atr_period = int(atr_period)
        self._atr_risk_fraction = float(atr_risk_fraction_of_nav)
        self._atr_max_alloc_fraction = float(atr_max_allocation_fraction_of_nav)
        self._chandelier_mult = float(chandelier_atr_multiple)
        self._highest_close_since_entry: float | None = None
        self._atr: pd.Series | None = None
        if self._use_atr_position_sizing:
            if high_series is None or low_series is None:
                raise ValueError("Stage 7 ATR sizing requires high_series and low_series.")
            # compute ATR on aligned H/L/C calendar (same index as closes)
            hi = high_series.copy()
            lo = low_series.copy()
            hi.index = pd.to_datetime(hi.index).normalize()
            lo.index = pd.to_datetime(lo.index).normalize()
            hi = hi.sort_index().reindex(self._closes.index).astype(float).fillna(self._closes)
            lo = lo.sort_index().reindex(self._closes.index).astype(float).fillna(self._closes)
            self._atr = _atr_from_hlc(hi, lo, self._closes, period=self._atr_period)
        # M2.1: build next-day open series (index T holds T+1's open price via shift(-1))
        self._use_next_day_open = bool(use_next_day_open)
        self._next_open: pd.Series | None = None
        if self._use_next_day_open and open_series is not None:
            op = open_series.copy()
            op.index = pd.to_datetime(op.index).normalize()
            op = op.sort_index().reindex(self._closes.index).astype(float)
            self._next_open = op.shift(-1)

        # M3.1: correlation gate state
        self._corr_data: dict[str, pd.Series] | None = corr_data
        self._corr_threshold = float(corr_threshold)
        self._corr_window = int(corr_window)
        self._my_symbol = str(self._instrument.name) if hasattr(self._instrument, "name") else ""

        if self._fee_model not in {"futu_us_stock", "none"}:
            raise ValueError(f"Unsupported fee_model={self._fee_model!r}; expected 'futu_us_stock' or 'none'.")

    def _rolling_corr_gate(self, i: int, backtest) -> bool:
        """Return True (block entry) when 30-day rolling correlation to any held position >= threshold."""
        if self._corr_data is None or backtest is None:
            return False
        holdings = getattr(backtest, "holdings", {})
        held_syms = [
            str(getattr(inst, "name", ""))
            for inst, qty in holdings.items()
            if float(qty) > 1e-9 and str(getattr(inst, "name", "")) != self._my_symbol
        ]
        if not held_syms:
            return False
        w = self._corr_window
        # need w+1 prices to get w returns; bail out if not enough history
        if i < w + 1:
            return False
        my_prices = self._closes.iloc[i - w - 1 : i]
        my_ret = my_prices.pct_change().dropna()
        for sym in held_syms:
            other_s = self._corr_data.get(sym)
            if other_s is None:
                continue
            other_s = other_s.reindex(self._closes.index).ffill()
            other_prices = other_s.iloc[i - w - 1 : i]
            other_ret = other_prices.pct_change().dropna()
            combined = pd.concat([my_ret, other_ret], axis=1).dropna()
            if len(combined) < 10:
                continue
            corr = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
            if math.isfinite(corr) and abs(corr) >= self._corr_threshold:
                return True
        return False

    def get_trigger_times(self) -> list:
        return [self._eod]

    # customization: Stage 7 sizing needs current NAV (engine values at daily mark-to-market, stored in performance)
    def _latest_nav_usd(self, backtest) -> float:
        if backtest is None:
            return float("nan")
        perf = getattr(backtest, "performance", None)
        try:
            if perf is not None and len(perf.dropna()):
                return float(perf.dropna().iloc[-1])
        except Exception:
            pass
        try:
            return float(getattr(backtest, "initial_value", float("nan")))
        except Exception:
            return float("nan")

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

        # M2.1: fill at next-day open when available; fall back to close_t for last bar or missing data
        _fill_ref = close_t
        if self._use_next_day_open and self._next_open is not None and i < len(self._next_open):
            _raw_open = self._next_open.iloc[i]
            if not pd.isna(_raw_open) and float(_raw_open) > self._eps:
                _fill_ref = float(_raw_open)

        buy_px_fee = _fill_ref * (1.0 + slip)
        sell_px_fee = _fill_ref * (1.0 - slip)
        # pass pre-computed fill price to orders when using next-day open (bypasses data_handler lookup)
        _override_buy = buy_px_fee if self._use_next_day_open else None
        _override_sell = sell_px_fee if self._use_next_day_open else None

        # resolve ADX once per bar (used by both regime modes and legacy gate)
        adx_t: float = float("nan")
        if self._adx is not None:
            adx_raw = self._adx.iloc[i] if i < len(self._adx) else float("nan")
            adx_t = float(adx_raw) if not pd.isna(adx_raw) else float("nan")

        if pos <= self._eps:
            self._purchase_price = None
            self._highest_close_since_entry = None
            self._entry_mode = None

            buy_signal = False
            new_entry_mode: str | None = None

            if self._use_regime_switching:
                if math.isfinite(adx_t):
                    if adx_t > self._adx_trend_min:
                        # Mode A (Trend): SMA crossover entry
                        if close_prev <= sma_prev and close_t > sma_t:
                            buy_signal = True
                            new_entry_mode = "trend"
                    elif adx_t < self._adx_mean_rev_max:
                        # Mode B (Asymmetric Mean Reversion): Lower Band touch AND RSI(14)<30
                        if self._lower_band is not None:
                            lb_raw = self._lower_band.iloc[i] if i < len(self._lower_band) else float("nan")
                            lb_t = float(lb_raw) if not pd.isna(lb_raw) else float("nan")
                            rsi_t = float("nan")
                            if self._rsi is not None:
                                rsi_raw = self._rsi.iloc[i] if i < len(self._rsi) else float("nan")
                                rsi_t = float(rsi_raw) if not pd.isna(rsi_raw) else float("nan")
                            if math.isfinite(lb_t) and close_t <= lb_t and math.isfinite(rsi_t) and rsi_t < 30.0:
                                buy_signal = True
                                new_entry_mode = "mean_rev"
                    # ADX in dead zone [adx_mean_rev_max, adx_trend_min]: no new entries
            else:
                # original logic: SMA crossover + ADX > adx_buy_min gate
                if close_prev <= sma_prev and close_t > sma_t:
                    if self._adx is not None:
                        if not (adx_t == adx_t) or adx_t <= self._adx_buy_min:
                            return []
                    buy_signal = True

            if not buy_signal:
                return []
            if close_t <= self._eps:
                return []
            # M3.1: block entry when too correlated to an existing position
            if self._rolling_corr_gate(i, backtest):
                return []

            # position sizing: ATR inverse-vol (Stage 7+) or fixed dollar (Stage 6)
            if self._use_atr_position_sizing:
                if self._atr is None:
                    return []
                atr_raw = self._atr.iloc[i] if i < len(self._atr) else float("nan")
                atr_t = float(atr_raw) if not pd.isna(atr_raw) else float("nan")
                if not math.isfinite(atr_t) or atr_t <= self._eps:
                    return []
                nav = self._latest_nav_usd(backtest)
                if not math.isfinite(nav) or nav <= self._eps:
                    return []
                risk_usd = nav * self._atr_risk_fraction
                shares_float = risk_usd / atr_t
                qty = int(math.floor(shares_float))
                # cap max allocation to avoid over-sizing in low-volatility periods
                max_shares = int(math.floor((nav * self._atr_max_alloc_fraction) / _fill_ref))
                qty = int(min(qty, max_shares))
            else:
                if self._target_usd <= self._eps:
                    return []
                qty = int(math.floor(self._target_usd / _fill_ref))
            if qty <= 0:
                return []

            self._purchase_price = buy_px_fee
            self._highest_close_since_entry = close_t
            self._entry_mode = new_entry_mode
            orders.append(
                OrderAtMarketWithSlippage(
                    self._instrument,
                    float(qty),
                    state,
                    state,
                    "MACrossover",
                    slippage_fraction=slip,
                    override_fill_price=_override_buy,
                )
            )
            fee = self._execution_fee_usd(shares=float(qty), price=buy_px_fee, side="buy")
            if fee > self._eps:
                orders.append(OrderCost("USD", -abs(fee), "MACrossover", state))
        else:
            # Chandelier trailing stop (Stage 7+): HighestCloseSinceEntry - chandelier_mult * ATR
            chandelier_hit = False
            if self._use_atr_position_sizing:
                if self._highest_close_since_entry is None:
                    self._highest_close_since_entry = close_t
                else:
                    self._highest_close_since_entry = max(float(self._highest_close_since_entry), close_t)
                if self._atr is not None:
                    atr_raw = self._atr.iloc[i] if i < len(self._atr) else float("nan")
                    atr_t = float(atr_raw) if not pd.isna(atr_raw) else float("nan")
                    if math.isfinite(atr_t) and atr_t > self._eps and self._highest_close_since_entry is not None:
                        stop_level = float(self._highest_close_since_entry) - (self._chandelier_mult * atr_t)
                        if close_t <= stop_level:
                            chandelier_hit = True

            # Stage 6 hard stop below purchase price (only when not using ATR sizing)
            stop_hit = (
                (not self._use_atr_position_sizing)
                and self._purchase_price is not None
                and close_t <= self._purchase_price * (1.0 - self._stop_loss_fraction)
            )

            # regime-aware exit: mean reversion exits at Upper Bollinger Band (asymmetric target); trend exits below SMA
            if self._use_regime_switching and self._entry_mode == "mean_rev":
                if self._upper_band is not None:
                    ub_raw = self._upper_band.iloc[i] if i < len(self._upper_band) else float("nan")
                    ub_t = float(ub_raw) if not pd.isna(ub_raw) else float("nan")
                    trend_exit = math.isfinite(ub_t) and close_t >= ub_t
                else:
                    trend_exit = close_t >= sma_t
            else:
                trend_exit = close_t < sma_t

            if chandelier_hit or stop_hit or trend_exit:
                q = -pos
                if abs(q) > self._eps:
                    self._purchase_price = None
                    self._highest_close_since_entry = None
                    self._entry_mode = None
                    orders.append(
                        OrderAtMarketWithSlippage(
                            self._instrument,
                            q,
                            state,
                            state,
                            "MACrossover",
                            slippage_fraction=slip,
                            override_fill_price=_override_sell,
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


# customization: O/H/L/C from SQLite for ADX(14) and M2.1 next-day open execution
def _load_symbol_ohlc(db_path: Path, symbol: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT date, open_price, high_price, low_price, close_price FROM market_history WHERE symbol = ? ORDER BY date",
            conn,
            params=(symbol,),
        )
    idx = pd.to_datetime(df["date"]).dt.normalize()
    close = pd.to_numeric(df["close_price"], errors="coerce")
    high = pd.to_numeric(df["high_price"], errors="coerce").fillna(close)
    low = pd.to_numeric(df["low_price"], errors="coerce").fillna(close)
    open_ = pd.to_numeric(df["open_price"], errors="coerce").fillna(close)
    hi = pd.Series(high.to_numpy(), index=idx, name="high_price").sort_index()
    lo = pd.Series(low.to_numpy(), index=idx, name="low_price").sort_index()
    cl = pd.Series(close.to_numpy(), index=idx, name="close_price").sort_index()
    op = pd.Series(open_.to_numpy(), index=idx, name="open_price").sort_index()
    return hi, lo, cl, op


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


# Stage 9: Wilder-style RSI from close prices — used as exhaustion filter on mean-reversion entries
def _rsi_from_close(close: pd.Series, *, period: int = 14) -> pd.Series:
    c = close.astype(float).sort_index()
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.reindex(c.index)


# customization: Stage 7 — ATR (simple moving average of True Range) for inverse-vol sizing & Chandelier stops
def _atr_from_hlc(high: pd.Series, low: pd.Series, close: pd.Series, *, period: int = 20) -> pd.Series:
    h = high.astype(float).sort_index()
    lo = low.astype(float).sort_index()
    c = close.astype(float).sort_index()
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(int(period), min_periods=int(period)).mean()
    return atr.reindex(c.index)


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


def _win_loss_ratio_from_backtest(backtest) -> float:
    """Average winning trade / average losing trade (absolute value) from closed round-trips."""
    try:
        ledger = backtest.trade_ledger()
    except Exception:
        return float("nan")
    if ledger is None or ledger.empty or "Trade PnL" not in ledger.columns:
        return float("nan")
    pnl = ledger.loc[ledger["Status"] == "closed", "Trade PnL"].dropna().astype(float)
    if pnl.empty:
        return float("nan")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if losses.empty:
        return float("inf") if not wins.empty else float("nan")
    if wins.empty:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


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
    symbols: list[str] | None = None,
) -> tuple[DataManager, dict[str, pd.Series], dict[str, EqStock], dict[str, tuple[pd.Series, pd.Series, pd.Series]]]:
    """Build a DataManager for ``symbols`` (defaults to all ``_TICKERS``)."""
    if symbols is None:
        symbols = list(_TICKERS)
    ensure_market_db(db_path)

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
    # customization: Stage 7 — ATR sizing + Chandelier stop
    use_atr_position_sizing: bool = False,
    use_regime_switching: bool = False,
    adx_trend_min: float = 25.0,
    hlc_by_symbol: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] | None = None,
    use_next_day_open: bool = False,
    # M3.1 correlation gate (pass close_by_symbol dict to enable; None disables)
    corr_data: dict[str, pd.Series] | None = None,
    corr_threshold: float = 0.70,
    corr_window: int = 30,
):
    triggers = []
    for symbol in instruments.keys():
        hi = lo = op = None
        if hlc_by_symbol is not None and symbol in hlc_by_symbol:
            hi, lo, _cl, op = hlc_by_symbol[symbol]
        triggers.append(
            MACrossoverEODTrigger(
                instruments[symbol],
                close_by_symbol[symbol],
                target_usd_allocation=target_usd_allocation,
                sma_window=int(sma_window),
                fee_model=str(fee_model),
                slippage_fraction=float(slippage_fraction),
                stop_loss_fraction=float(stop_loss_fraction),
                use_atr_position_sizing=bool(use_atr_position_sizing),
                use_regime_switching=bool(use_regime_switching),
                adx_trend_min=float(adx_trend_min),
                high_series=hi,
                low_series=lo,
                open_series=op if use_next_day_open else None,
                use_next_day_open=use_next_day_open,
                corr_data=corr_data,
                corr_threshold=corr_threshold,
                corr_window=corr_window,
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
    _ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
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
        plt.savefig(_CHARTS_DIR / f"backtest_result_{_ts}.png", dpi=150)
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
    _ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
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

    # Stage 6 vs Stage 7 vs Stage 8 (2025 OOS) comparison
    print("\n=== Stage 6 vs Stage 7 vs Stage 8 (2025 OOS) — Regime-Switching Delta ===")
    bt25_s6 = bt25
    nav25_s6 = nav25
    # Stage 7: ATR sizing + Chandelier stop; ADX>20 gate
    bt25_s7, _s7_summary, nav25_s7, _s7_sh = _run_single_sma_window(
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
        use_atr_position_sizing=True,
        hlc_by_symbol=hlc_by_symbol,
    )
    # Stage 8: ADX gear-shifter — trend SMA crossover (ADX>25) vs mean-reversion Bollinger (ADX<20)
    bt25_s8, _s8_summary, nav25_s8, _s8_sh = _run_single_sma_window(
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
        use_atr_position_sizing=True,
        use_regime_switching=True,
        hlc_by_symbol=hlc_by_symbol,
    )
    _tr_s6, mdd_s6 = _nav_total_return_and_max_drawdown(nav25_s6)
    _tr_s7, mdd_s7 = _nav_total_return_and_max_drawdown(nav25_s7)
    _tr_s8, mdd_s8 = _nav_total_return_and_max_drawdown(nav25_s8)
    pf_s6 = _profit_factor_from_backtest(bt25_s6)
    pf_s7 = _profit_factor_from_backtest(bt25_s7)
    pf_s8 = _profit_factor_from_backtest(bt25_s8)
    report = pd.DataFrame(
        [
            {"Stage": "6 (Fixed)", "Year": "2025", "Max_Drawdown_pct": abs(float(mdd_s6)), "Profit_Factor": float(pf_s6)},
            {"Stage": "7 (ATR Defensive)", "Year": "2025", "Max_Drawdown_pct": abs(float(mdd_s7)), "Profit_Factor": float(pf_s7)},
            {"Stage": "8 (Regime Switching)", "Year": "2025", "Max_Drawdown_pct": abs(float(mdd_s8)), "Profit_Factor": float(pf_s8)},
        ]
    )
    print(report.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    pf_s8_val = float(pf_s8)
    pf_target_met = math.isfinite(pf_s8_val) and pf_s8_val > 1.0
    print(f"\nStage 8 Profit Factor > 1.0: {'YES' if pf_target_met else 'NO'} (value: {pf_s8_val:.6f})")

    # Stage 9: 2025 OOS Win/Loss Ratio for baseline Stage 8 run (ADX threshold=25)
    wl_s8 = _win_loss_ratio_from_backtest(bt25_s8)
    wl_s8_s = f"{wl_s8:.4f}" if math.isfinite(wl_s8) else ("inf (no losing trades)" if wl_s8 == float("inf") else "N/A")
    print(f"Stage 8 2025 OOS Win/Loss Ratio (avg win / avg loss): {wl_s8_s}")

    # Stage 9: Hysteresis sweep — ADX Trend Threshold across 25, 30, 35 (2025 OOS)
    print("\n=== Stage 9: ADX Trend Threshold Hysteresis Sweep (2025 OOS) ===")
    sweep9_rows: list[dict] = []
    for adx_thresh in [25, 30, 35]:
        bt9, _, nav9, sh9 = _run_single_sma_window(
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
            use_atr_position_sizing=True,
            use_regime_switching=True,
            adx_trend_min=float(adx_thresh),
            hlc_by_symbol=hlc_by_symbol,
        )
        tr9, mdd9 = _nav_total_return_and_max_drawdown(nav9)
        pf9 = _profit_factor_from_backtest(bt9)
        wl9 = _win_loss_ratio_from_backtest(bt9)
        sweep9_rows.append({
            "ADX_Trend_Threshold": adx_thresh,
            "Sharpe": sh9,
            "Total_Return_pct": tr9,
            "Max_Drawdown_pct": abs(float(mdd9)),
            "Profit_Factor": pf9,
            "Win_Loss_Ratio": wl9,
        })
    sweep9_df = pd.DataFrame(sweep9_rows)
    print(sweep9_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("(Win/Loss Ratio = avg winning trade / avg losing trade; higher is better for asymmetric exits)")

    # M2.1: Next-Day Open Execution — Reality Check comparison vs M1.9 Close Execution (2025 OOS)
    print("\n=== M2.1: Phase 2 Reality Check — Next-Day Open vs Close Execution (2025 OOS) ===")
    # M1.9 baseline reuses bt25_s8 (Stage 9 regime switching, ADX=25, close execution)
    bt_m21, _, nav_m21, sh_m21 = _run_single_sma_window(
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
        use_atr_position_sizing=True,
        use_regime_switching=True,
        adx_trend_min=25.0,
        hlc_by_symbol=hlc_by_symbol,
        use_next_day_open=True,
    )
    tr_m19, _ = _nav_total_return_and_max_drawdown(nav25_s8)
    tr_m21, _ = _nav_total_return_and_max_drawdown(nav_m21)
    pf_m21 = _profit_factor_from_backtest(bt_m21)
    comparison_m21 = pd.DataFrame([
        {"Milestone": "M1.9 (Close Execution)", "Annualized_Sharpe": _s8_sh,
         "Total_Return_pct": tr_m19, "Profit_Factor": pf_s8},
        {"Milestone": "M2.1 (Next-Day Open)",   "Annualized_Sharpe": sh_m21,
         "Total_Return_pct": tr_m21, "Profit_Factor": pf_m21},
    ])
    print(comparison_m21.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    pf_m21_val = float(pf_m21)
    if math.isfinite(pf_m21_val):
        litmus = (
            "PASS — strategy remains viable after execution lag"
            if pf_m21_val >= 0.90
            else "CAUTION — strategy may be scalping overnight gaps not capturable in live trading"
        )
        print(f"\nM2.1 Litmus Test (Profit Factor >= 0.90): {litmus} (value: {pf_m21_val:.4f})")

    _save_stability_heatmap(sweep_df, str(_CHARTS_DIR / f"stability_heatmap_{_ts}.png"))
    print(f"\nWrote charts/stability_heatmap_{_ts}.png (2024 Sharpe vs SMA window).")

    if len(nav25.dropna()):
        nav_plot = nav25.dropna().astype(float)
        plt.figure(figsize=(10, 5))
        plt.plot(nav_plot.index, nav_plot.values, linewidth=1.5)
        plt.title(f"2025 OOS equity (SMA={best_w}, Sharpe={sh25:.4f})")
        plt.xlabel("Date")
        plt.ylabel("Total NAV")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(_CHARTS_DIR / f"backtest_result_{_ts}.png", dpi=150)
        plt.close()

    print("\nDone (Stage 9, no GsSession).")


def _compute_correlation_matrix(
    close_by_symbol: dict[str, pd.Series],
    start: dt.date,
    end: dt.date,
) -> "pd.DataFrame":
    """Pearson correlation matrix of daily returns over [start, end] (inclusive)."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames: dict[str, pd.Series] = {}
    for sym, s in close_by_symbol.items():
        s2 = s.astype(float).dropna()
        mask = (s2.index >= start_ts) & (s2.index <= end_ts)
        frames[sym] = s2[mask].pct_change()
    ret_df = pd.DataFrame(frames).dropna(how="all")
    return ret_df.corr(method="pearson").round(4)


def run_m31_portfolio(
    db_path: Path,
    *,
    sma_window: int = 20,
    target_usd_allocation: float = 25_000.0,
    fee_model: str = "futu_us_stock",
    initial_value: float | None = None,
    slippage_fraction: float = DEFAULT_SLIPPAGE_FRACTION,
    corr_threshold: float = 0.70,
    corr_window: int = 30,
) -> None:
    """
    M3.1 — Multi-Symbol Correlation & Scaling.

    1. Downloads & ingests 10-ticker universe (parallel via ThreadPoolExecutor).
    2. Prints Pearson Correlation Matrix (2024 in-sample daily returns).
    3. Runs combined portfolio backtest (2025 OOS, M2.1 execution, correlation gate).
    4. Runs per-ticker individual backtests (no correlation gate) for baseline MDD.
    5. Prints Diversification Benefit = avg(individual MDD) − portfolio MDD.
    """
    iv = float(initial_value) if initial_value is not None else max(
        1_000_000.0, 3.0 * float(target_usd_allocation) * len(_TICKERS)
    )
    _ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    start_2024 = dt.date(2024, 1, 1)
    end_2024 = dt.date(2024, 12, 31)
    start_2025 = dt.date(2025, 1, 1)
    end_2025 = dt.date(2025, 12, 31)

    print(f"=== M3.1: Multi-Symbol Portfolio — {len(_TICKERS)}-ticker universe ===")
    print(f"Tickers: {', '.join(_TICKERS)}\n")

    # Build data manager (triggers parallel yfinance download for all tickers)
    data_manager, close_by_symbol, instruments, hlc_by_symbol = _build_data_manager(db_path)

    # ── 1. Pearson Correlation Matrix (2024 in-sample) ─────────────────────────
    print("=== Pearson Correlation Matrix (2024 in-sample daily returns) ===")
    corr_matrix = _compute_correlation_matrix(close_by_symbol, start_2024, end_2024)
    print(corr_matrix.to_string(float_format=lambda x: f"{x:+.3f}"))
    print()

    # ── 2. Combined portfolio backtest (2025 OOS, correlation gate ON) ─────────
    print("=== Combined Portfolio Backtest (2025 OOS, M2.1 execution, corr gate ON) ===")
    bt_port, _, nav_port, sh_port = _run_single_sma_window(
        start=start_2025,
        end=end_2025,
        data_manager=data_manager,
        close_by_symbol=close_by_symbol,
        instruments=instruments,
        sma_window=sma_window,
        target_usd_allocation=float(target_usd_allocation),
        fee_model=fee_model,
        initial_value=iv,
        slippage_fraction=slippage_fraction,
        use_atr_position_sizing=True,
        use_regime_switching=True,
        adx_trend_min=25.0,
        hlc_by_symbol=hlc_by_symbol,
        use_next_day_open=True,
        corr_data=close_by_symbol,
        corr_threshold=corr_threshold,
        corr_window=corr_window,
    )
    tr_port, mdd_port = _nav_total_return_and_max_drawdown(nav_port)
    pf_port = _profit_factor_from_backtest(bt_port)

    # ── 3. Individual ticker backtests (no corr gate, shared data_manager) ─────
    print("\n=== Individual Ticker Performance (2025 OOS, no correlation gate) ===")
    ind_iv = max(100_000.0, 2.5 * float(target_usd_allocation))
    individual_rows: list[dict] = []
    for sym in _TICKERS:
        _, _, nav_i, sh_i = _run_single_sma_window(
            start=start_2025,
            end=end_2025,
            data_manager=data_manager,
            close_by_symbol={sym: close_by_symbol[sym]},
            instruments={sym: instruments[sym]},
            sma_window=sma_window,
            target_usd_allocation=float(target_usd_allocation),
            fee_model=fee_model,
            initial_value=ind_iv,
            slippage_fraction=slippage_fraction,
            use_atr_position_sizing=True,
            use_regime_switching=True,
            adx_trend_min=25.0,
            hlc_by_symbol={sym: hlc_by_symbol[sym]},
            use_next_day_open=True,
        )
        tr_i, mdd_i = _nav_total_return_and_max_drawdown(nav_i)
        mdd_i_abs = abs(float(mdd_i)) if math.isfinite(float(mdd_i)) else float("nan")
        individual_rows.append(
            {"Symbol": sym, "Sharpe": sh_i, "Total_Return_pct": tr_i, "Max_Drawdown_pct": mdd_i_abs}
        )
        sharpe_s = f"{sh_i:.4f}" if math.isfinite(sh_i) else "N/A"
        print(f"  {sym:5s}  Sharpe={sharpe_s:>8}  Return={tr_i:+.2f}%  MaxDD={mdd_i_abs:.2f}%")

    ind_df = pd.DataFrame(individual_rows)
    avg_ind_mdd = float(ind_df["Max_Drawdown_pct"].dropna().mean())
    port_mdd_abs = abs(float(mdd_port)) if math.isfinite(float(mdd_port)) else float("nan")
    divers_benefit = avg_ind_mdd - port_mdd_abs

    # ── 4. Final report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== M3.1 Combined Portfolio Report (2025 OOS) ===")
    print("=" * 60)
    sharpe_s = f"{sh_port:.4f}" if math.isfinite(sh_port) else "N/A"
    pf_s = f"{pf_port:.4f}" if math.isfinite(pf_port) else ("inf" if pf_port == float("inf") else "N/A")
    print(f"Portfolio Annualized Sharpe :  {sharpe_s}")
    print(f"Portfolio Total Return      :  {tr_port:+.4f}%")
    print(f"Portfolio Max Drawdown      :  {port_mdd_abs:.4f}%")
    print(f"Portfolio Profit Factor     :  {pf_s}")
    print()
    print(f"Avg Individual Max Drawdown :  {avg_ind_mdd:.4f}%")
    print(f"Portfolio Max Drawdown      :  {port_mdd_abs:.4f}%")
    sign = "+" if divers_benefit >= 0 else ""
    print(f"Diversification Benefit     :  {sign}{divers_benefit:.4f}%  (positive = correlation filter reduced drawdown)")
    print(f"\nCorrelation gate: 30-day rolling Pearson threshold = {corr_threshold:.0%}")
    print(f"Universe: {', '.join(_TICKERS)}")

    # Plot portfolio equity curve
    if len(nav_port.dropna()):
        nav_plot = nav_port.dropna().astype(float)
        plt.figure(figsize=(12, 5))
        plt.plot(nav_plot.index, nav_plot.values, linewidth=1.5, color="steelblue", label="Portfolio Total NAV")
        plt.title(
            f"M3.1 Combined Portfolio Equity Curve — 2025 OOS\n"
            f"{len(_TICKERS)} tickers, corr gate {corr_threshold:.0%}, SMA={sma_window}"
        )
        plt.xlabel("Date")
        plt.ylabel("Total NAV (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = _CHARTS_DIR / f"m31_portfolio_{_ts}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\nWrote {out}")

    print("\nDone (M3.1, no GsSession).")


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
        help="Starting portfolio NAV in USD",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=31,
        choices=[6, 9, 10, 31],
        help=(
            "Stage to run: 6/9/10 = legacy Stage 6–M2.1 robustness suite (2-ticker); "
            "31 = M3.1 multi-symbol portfolio with correlation engine (default)"
        ),
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
        if args.stage == 31:
            run_m31_portfolio(
                args.db.resolve(),
                initial_value=args.initial_value,
            )
        else:
            # stages 6 / 9 / 10 — legacy 2-ticker robustness suite
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
