"""
M5.2 — Multi-Logic Strategy Tournament

Compares three trading philosophies on 1810.HK (or any HK target):
  TechnicalOnly      — EMA(20/50) crossover + RSI(14) < 70 filter
  SentimentAugmented — TechnicalOnly + sentiment_score > 0.5 gate (from sentiment API)
  MarketNeutralAlpha — trades only when alpha_velocity > 0 (ignores market direction)

Sizing: ATR-based risk budget, HKEX 200-share lot rounding, 5 bps slippage.
Output:
  experiments/outputs/artifacts/tournament_<ts>.png   — equity-curve chart
  experiments/outputs/artifacts/comparison_report.csv — summary metrics

Usage:
    python research/experiments/tournament_optimizer.py
    python research/experiments/tournament_optimizer.py --target 1024.HK --start 2025-01-01 --end 2026-01-01
    python research/experiments/tournament_optimizer.py --sentiment-fallback sentiment_fallback.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from gs_quant.timeseries.econometrics import beta as gsq_beta
from gs_quant.timeseries.helper import Window
from gs_quant.timeseries.technicals import (
    average_true_range,
    exponential_moving_average as gsq_ema,
    moving_average as gsq_ma,
    relative_strength_index as gsq_rsi,
)

from research.engines.beta_engine import BetaCalculator
from research.engines.output_manager import OutputManager

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_REPO_ROOT = Path(__file__).parent.parent.parent
_RISK_FREE_ANNUAL = float(os.getenv("RISK_FREE_ANNUAL", "0.02"))
_SLIPPAGE_BPS = int(os.getenv("SLIPPAGE_BPS", "5"))
_HK_LOT_SIZE = int(os.getenv("HK_LOT_SIZE", "200"))
_ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
_RISK_BUDGET_FRACTION = float(os.getenv("RISK_BUDGET_FRACTION", "0.01"))
_SENTIMENT_API_URL = os.getenv("SENTIMENT_API_URL", "http://192.168.31.208:8000")
_SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY", "quant_local")
_DATA_START = os.getenv("DATA_START", "2025-01-01")
_DATA_END_EXCLUSIVE = os.getenv("DATA_END_EXCLUSIVE", "2026-01-01")

# EMA periods → beta coefficients: β = (N-1)/(N+1)
_EMA_FAST_BETA = 19 / 21   # 20-period EMA
_EMA_SLOW_BETA = 49 / 51   # 50-period EMA


# ---------------------------------------------------------------------------
# Sentiment client
# ---------------------------------------------------------------------------

class SentimentClient:
    """Fetch sentiment scores from the REST API; fall back to CSV if unreachable."""

    def __init__(self, base_url: str = _SENTIMENT_API_URL, api_key: str = _SENTIMENT_API_KEY,
                 fallback_csv: Path | None = None):
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self._fallback_csv = fallback_csv

    def fetch(self, symbol: str, start: str, end: str) -> pd.Series:
        """Return a date-indexed Series of sentiment_score. Falls back to CSV on any error."""
        try:
            resp = requests.get(
                f"{self._base_url}/sentiment",
                params={"symbol": symbol, "start": start, "end": end},
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date")["sentiment_score"].rename("sentiment_score")
        except Exception as exc:
            print(f"  [SentimentClient] API unreachable ({exc}); trying CSV fallback.")

        if self._fallback_csv and self._fallback_csv.exists():
            df = pd.read_csv(self._fallback_csv, parse_dates=["date"])
            filtered = df[(df["symbol"] == symbol) & (df["date"] >= start) & (df["date"] < end)]
            return filtered.set_index("date")["sentiment_score"].rename("sentiment_score")

        print("  [SentimentClient] No fallback available — sentiment will be neutral (0).")
        return pd.Series(dtype=float, name="sentiment_score")


# ---------------------------------------------------------------------------
# Shared backtest helpers
# ---------------------------------------------------------------------------

def _lot_floor(shares: float, lot: int = _HK_LOT_SIZE) -> int:
    """Round down to nearest HKEX lot size."""
    return int(shares // lot) * lot


def _apply_slippage(price: float, is_buy: bool, bps: int = _SLIPPAGE_BPS) -> float:
    factor = 1 + bps / 10_000 if is_buy else 1 - bps / 10_000
    return price * factor


def _run_backtest(
    signals: pd.Series,          # 1 = long, 0 = flat
    prices: pd.Series,
    atr: pd.Series,
    initial_nav: float = 10_000.0,
) -> tuple[pd.Series, list[float]]:
    """
    Vectorized backtest loop.
    Position size is computed fresh at each entry using ATR-based risk budget.
    Returns (nav_series, trade_pnl_list).
    """
    nav = initial_nav
    shares_held = 0
    entry_price = 0.0
    nav_series: dict[pd.Timestamp, float] = {}
    trade_pnls: list[float] = []

    common = signals.index.intersection(prices.index).intersection(atr.index)
    signals = signals.reindex(common)
    prices = prices.reindex(common)
    atr = atr.reindex(common)

    prev_signal = 0
    for date in common:
        price = prices[date]
        sig = int(signals[date]) if not math.isnan(signals[date]) else 0
        atr_val = atr[date] if not math.isnan(atr[date]) else price * 0.02

        # Entry
        if sig == 1 and prev_signal == 0 and shares_held == 0:
            fill = _apply_slippage(price, is_buy=True)
            risk_usd = nav * _RISK_BUDGET_FRACTION
            raw_shares = risk_usd / atr_val if atr_val > 0 else 0
            shares_held = _lot_floor(raw_shares)
            entry_price = fill
            if shares_held > 0:
                nav -= shares_held * fill

        # Exit
        elif sig == 0 and prev_signal == 1 and shares_held > 0:
            fill = _apply_slippage(price, is_buy=False)
            proceeds = shares_held * fill
            pnl = proceeds - shares_held * entry_price
            trade_pnls.append(pnl)
            nav += proceeds
            shares_held = 0
            entry_price = 0.0

        # Mark-to-market
        nav_series[date] = nav + shares_held * price
        prev_signal = sig

    # Force close on last bar if still holding
    if shares_held > 0:
        last_date = common[-1]
        fill = _apply_slippage(float(prices.iloc[-1]), is_buy=False)
        proceeds = shares_held * fill
        pnl = proceeds - shares_held * entry_price
        trade_pnls.append(pnl)
        nav_series[last_date] = nav + proceeds

    return pd.Series(nav_series, name="nav"), trade_pnls


def _metrics(nav: pd.Series, trade_pnls: list[float], benchmark_returns: pd.Series,
             label: str) -> dict:
    """Compute tournament metrics for one strategy."""
    if nav.empty or len(nav) < 2:
        return {"strategy": label, **{k: float("nan") for k in
                ["ann_return", "sharpe", "max_drawdown", "win_rate", "profit_factor",
                 "alpha_contribution"]}}

    daily_ret = nav.pct_change().dropna()
    total_return = (nav.iloc[-1] / nav.iloc[0] - 1)
    n_days = len(daily_ret)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    excess = daily_ret - _RISK_FREE_ANNUAL / 252
    sharpe = (excess.mean() / excess.std() * math.sqrt(252)) if excess.std() > 0 else float("nan")

    roll_max = nav.cummax()
    drawdown = (nav - roll_max) / roll_max
    max_dd = float(drawdown.min())

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else float("nan")
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Market Alpha Contribution: regress strategy daily returns on benchmark returns
    common = daily_ret.index.intersection(benchmark_returns.index)
    if len(common) > 20:
        s_ret = daily_ret.reindex(common)
        b_ret = benchmark_returns.reindex(common)
        beta_val = gsq_beta(
            (1 + s_ret).cumprod(),
            (1 + b_ret).cumprod(),
            Window(len(common), len(common)),
            prices=True,
        )
        beta_scalar = float(beta_val.dropna().iloc[-1]) if not beta_val.dropna().empty else float("nan")
        alpha_contrib = 1.0 - abs(beta_scalar) if math.isfinite(beta_scalar) else float("nan")
    else:
        alpha_contrib = float("nan")

    return {
        "strategy": label,
        "ann_return": round(ann_return * 100, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 2) if math.isfinite(win_rate) else float("nan"),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else float("inf"),
        "alpha_contribution": round(alpha_contrib * 100, 2) if math.isfinite(alpha_contrib) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    def __init__(self, label: str):
        self.label = label

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.Series,
        atr: pd.Series,
        sentiment: pd.Series,
        alpha_velocity: pd.Series,
    ) -> pd.Series:
        ...

    def run(self, prices: pd.Series, atr: pd.Series, sentiment: pd.Series,
            alpha_velocity: pd.Series, benchmark_returns: pd.Series,
            initial_nav: float = 10_000.0) -> tuple[dict, pd.Series]:
        signals = self.generate_signals(prices, atr, sentiment, alpha_velocity)
        nav, trade_pnls = _run_backtest(signals, prices, atr, initial_nav)
        return _metrics(nav, trade_pnls, benchmark_returns, self.label), nav


class TechnicalOnly(BaseStrategy):
    """EMA(20) / EMA(50) crossover with RSI(14) < 70 overbought filter."""

    def __init__(self):
        super().__init__("TechnicalOnly")

    def generate_signals(self, prices, atr, sentiment, alpha_velocity) -> pd.Series:
        ema_fast = gsq_ema(prices, _EMA_FAST_BETA)
        ema_slow = gsq_ema(prices, _EMA_SLOW_BETA)
        rsi = gsq_rsi(prices, 14)

        bullish = (ema_fast > ema_slow) & (rsi < 70)
        return bullish.astype(int).rename("signal")


class SentimentAugmented(BaseStrategy):
    """TechnicalOnly + sentiment_score > 0.5 gate at entry."""

    def __init__(self):
        super().__init__("SentimentAugmented")
        self._technical = TechnicalOnly()

    def generate_signals(self, prices, atr, sentiment, alpha_velocity) -> pd.Series:
        tech_signals = self._technical.generate_signals(prices, atr, sentiment, alpha_velocity)

        if sentiment.empty:
            print(f"  [{self.label}] No sentiment data — running as TechnicalOnly.")
            return tech_signals

        # Reindex sentiment to price dates; forward-fill (sentiment published after market close)
        sent_aligned = sentiment.reindex(prices.index).ffill().fillna(0.0)
        return (tech_signals & (sent_aligned > 0.5)).astype(int).rename("signal")


class MarketNeutralAlpha(BaseStrategy):
    """
    Trades purely on alpha velocity — long when MA(5) > MA(20) of cumulative residual return.
    Ignores absolute market direction.
    """

    def __init__(self):
        super().__init__("MarketNeutralAlpha")

    def generate_signals(self, prices, atr, sentiment, alpha_velocity) -> pd.Series:
        if alpha_velocity.empty:
            print(f"  [{self.label}] No alpha velocity — no signals generated.")
            return pd.Series(0, index=prices.index, name="signal")

        av_aligned = alpha_velocity.reindex(prices.index).ffill()
        return (av_aligned > 0).astype(int).rename("signal")


# ---------------------------------------------------------------------------
# Tournament engine
# ---------------------------------------------------------------------------

class TournamentEngine:

    def __init__(self, calc: BetaCalculator, sentiment_client: SentimentClient,
                 start: str, end: str, initial_nav: float = 10_000.0):
        self._calc = calc
        self._sentiment_client = sentiment_client
        self._start = start
        self._end = end
        self._initial_nav = initial_nav

    def run(self) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
        calc = self._calc
        prices = calc._ticker_prices
        benchmark_prices = calc._benchmark_prices
        alpha_velocity = calc.get_alpha_velocity()
        benchmark_returns = benchmark_prices.pct_change().dropna()

        # ATR requires OHLC — approximate with close-only (high=low=close for daily close series)
        # Using the close series directly: TR = |close_t - close_{t-1}|
        atr = average_true_range(prices, prices, prices, _ATR_WINDOW)

        sentiment = self._sentiment_client.fetch(
            calc.target_ticker, self._start, self._end
        )

        strategies: list[BaseStrategy] = [
            TechnicalOnly(),
            SentimentAugmented(),
            MarketNeutralAlpha(),
        ]

        rows: list[dict] = []
        navs: dict[str, pd.Series] = {}

        for strategy in strategies:
            print(f"  Running {strategy.label}…")
            metrics, nav = strategy.run(
                prices, atr, sentiment, alpha_velocity, benchmark_returns, self._initial_nav
            )
            rows.append(metrics)
            navs[strategy.label] = nav

        report = pd.DataFrame(rows).set_index("strategy")
        return report, navs

    def plot(self, report: pd.DataFrame, navs: dict[str, pd.Series],
             benchmark_prices: pd.Series, output_path: Path | None = None) -> Path:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OutputManager().artifact_path(f"tournament_{ts}.png")

        colors = {
            "TechnicalOnly": "#1f77b4",
            "SentimentAugmented": "#2ca02c",
            "MarketNeutralAlpha": "#9467bd",
        }

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        ax_equity = fig.add_subplot(gs[0, :])   # top row full width
        ax_dd = fig.add_subplot(gs[1, 0])
        ax_corr = fig.add_subplot(gs[1, 1])

        # Panel 1 — Equity curves + benchmark
        bench_norm = benchmark_prices / benchmark_prices.iloc[0] * self._initial_nav
        ax_equity.plot(bench_norm.index, bench_norm.values,
                       color="darkorange", linewidth=1.2, linestyle="--", label="HSTECH Benchmark", alpha=0.8)
        for label, nav in navs.items():
            if not nav.empty:
                ax_equity.plot(nav.index, nav.values, color=colors[label], linewidth=1.5, label=label)
        ax_equity.set_title(f"Cumulative Equity — {self._calc.target_ticker} ({self._start} → {self._end})", fontsize=12)
        ax_equity.set_ylabel("Portfolio Value ($)")
        ax_equity.legend(fontsize=9)
        ax_equity.grid(True, alpha=0.3)

        # Panel 2 — Drawdown
        for label, nav in navs.items():
            if not nav.empty:
                dd = (nav - nav.cummax()) / nav.cummax() * 100
                ax_dd.plot(dd.index, dd.values, color=colors[label], linewidth=1.2, label=label)
        ax_dd.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.05, color="red")
        ax_dd.set_title("Drawdown (%)")
        ax_dd.set_ylabel("%")
        ax_dd.legend(fontsize=8)
        ax_dd.grid(True, alpha=0.3)

        # Panel 3 — Returns correlation heatmap
        ret_df = pd.DataFrame({
            label: nav.pct_change().dropna()
            for label, nav in navs.items() if not nav.empty
        })
        if not ret_df.empty:
            corr = ret_df.corr()
            im = ax_corr.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
            ax_corr.set_xticks(range(len(corr.columns)))
            ax_corr.set_yticks(range(len(corr.index)))
            ax_corr.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=8)
            ax_corr.set_yticklabels(corr.index, fontsize=8)
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
            fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            ax_corr.set_title("Returns Correlation Matrix", fontsize=10)

        fig.suptitle(
            f"M5.2 Strategy Tournament: {self._calc.target_ticker} vs {self._calc.benchmark_ticker}",
            fontsize=13, fontweight="bold"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved → {output_path}")
        return output_path

    def save_report(self, report: pd.DataFrame, output_path: Path | None = None) -> Path:
        out = output_path or OutputManager().artifact_path("comparison_report.csv")
        report.to_csv(out)
        print(f"Report saved → {out}")
        return out

    def print_summary(self, report: pd.DataFrame) -> None:
        print("\n" + "=" * 70)
        print(f"  M5.2 Tournament Results — {self._calc.target_ticker} vs {self._calc.benchmark_ticker}")
        print("=" * 70)
        cols = ["ann_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "alpha_contribution"]
        headers = ["Ann Return%", "Sharpe", "Max DD%", "Win Rate%", "Profit Factor", "Alpha Contrib%"]
        print(f"  {'Strategy':<22} " + "  ".join(f"{h:>14}" for h in headers))
        print("  " + "-" * 66)
        for strategy, row in report.iterrows():
            def _fmt(v, is_pf=False):
                if is_pf:
                    return "inf" if v == float("inf") else f"{v:>14.4f}"
                return f"{v:>14.2f}" if math.isfinite(float(v)) else f"{'N/A':>14}"
            vals = [
                _fmt(row["ann_return"]),
                _fmt(row["sharpe"]),
                _fmt(row["max_drawdown"]),
                _fmt(row["win_rate"]),
                _fmt(row["profit_factor"], is_pf=True),
                _fmt(row["alpha_contribution"]),
            ]
            print(f"  {strategy:<22} " + "  ".join(vals))

        # Interpretation
        print("\n  Interpretation:")
        best_sharpe = report["sharpe"].idxmax() if report["sharpe"].notna().any() else None
        lowest_dd = report["max_drawdown"].idxmax() if report["max_drawdown"].notna().any() else None  # max_dd is negative
        if best_sharpe:
            print(f"  → Best risk-adjusted return:  {best_sharpe}")
        if lowest_dd:
            print(f"  → Lowest drawdown:            {lowest_dd}  (professional's choice)")

        # Correlation note
        print("\n  Note: check the correlation matrix in the chart.")
        print("  If SentimentAugmented ≈ TechnicalOnly correlation > 0.95,")
        print("  sentiment is not adding a new dimension — it is echoing price.")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Production sync stub
# ---------------------------------------------------------------------------

def sync_champion_to_core_db(metrics_dict: dict) -> None:
    """
    Stub: push the winning strategy's aggregated metadata to the Linux PostgreSQL server.

    Only the champion's summary payload travels over the API — raw scratch data (trade
    logs, interim CSVs, SQLite) stays on local Mac storage and is never uploaded.
    Replace the print statement with an authenticated HTTP POST to the core DB API
    when the production endpoint is ready.
    """
    print(
        f"[sync_champion_to_core_db] READY — champion payload staged for production DB push:\n"
        f"  {metrics_dict}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M5.2 Strategy Tournament — HK factor-neutral alpha")
    parser.add_argument("--target", default="1810.HK")
    parser.add_argument("--benchmark", default="3033.HK")
    parser.add_argument("--window", type=int, default=60, help="Rolling OLS window for beta (default 60)")
    parser.add_argument("--start", default=_DATA_START)
    parser.add_argument("--end", default=_DATA_END_EXCLUSIVE)
    parser.add_argument("--initial-nav", type=float, default=10_000.0)
    parser.add_argument("--sentiment-api", default=_SENTIMENT_API_URL)
    parser.add_argument("--sentiment-key", default=_SENTIMENT_API_KEY)
    parser.add_argument("--sentiment-fallback", type=Path, default=None,
                        help="CSV fallback path with columns: date, symbol, sentiment_score")
    args = parser.parse_args()

    calc = BetaCalculator(target_ticker=args.target, benchmark_ticker=args.benchmark, window=args.window)
    calc.load(start=args.start, end=args.end)
    calc.compute_rolling_beta()
    calc.compute_residual_returns()
    calc.compute_alpha_velocity()

    sentiment_client = SentimentClient(
        base_url=args.sentiment_api,
        api_key=args.sentiment_key,
        fallback_csv=args.sentiment_fallback,
    )

    engine = TournamentEngine(
        calc=calc,
        sentiment_client=sentiment_client,
        start=args.start,
        end=args.end,
        initial_nav=args.initial_nav,
    )

    om = OutputManager()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nRunning tournament…")
    report, navs = engine.run()
    engine.print_summary(report)
    engine.save_report(report, output_path=om.artifact_path("comparison_report.csv"))
    engine.plot(report, navs, calc._benchmark_prices,
                output_path=om.artifact_path(f"tournament_{ts}.png"))

    # Identify champion by Sharpe and push metadata stub to production DB
    if report["sharpe"].notna().any():
        champion = report["sharpe"].idxmax()
        champion_metrics = {"strategy": champion, **report.loc[champion].to_dict()}
        sync_champion_to_core_db(champion_metrics)


if __name__ == "__main__":
    main()
