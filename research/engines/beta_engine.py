"""
Module M5.1 — Residual Return (Beta) Engine

Strips the Hang Seng Tech Index (^HSTECH) market factor from Xiaomi (1810.HK)
daily returns to isolate idiosyncratic alpha via rolling OLS regression.

Usage:
    python beta_engine.py
    python research/beta_engine.py --target 1024.HK --start 2025-01-01 --end 2026-05-14 --window 60
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gs_quant.timeseries.econometrics import beta as gsq_beta
from gs_quant.timeseries.helper import Window
from gs_quant.timeseries.technicals import moving_average as gsq_ma

from research.engines.data_loader import HistoricalDataLoader

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_DATA_START = os.getenv("DATA_START", "2025-01-01")
_DATA_END_EXCLUSIVE = os.getenv("DATA_END_EXCLUSIVE", "2026-01-01")


class BetaCalculator:
    """
    Rolling OLS beta of a target ticker vs a benchmark, with residual return extraction.

    Residual return = R_ticker - (rolling_beta × R_benchmark)
    This strips the common market factor, leaving the idiosyncratic (alpha) component.
    """

    def __init__(self, target_ticker: str = "1810.HK", benchmark_ticker: str = "3033.HK", window: int = 60):
        self.target_ticker = target_ticker
        self.benchmark_ticker = benchmark_ticker
        self.window = window

        self._ticker_prices: pd.Series | None = None
        self._benchmark_prices: pd.Series | None = None
        self._rolling_beta: pd.Series | None = None
        self._residual_returns: pd.Series | None = None
        self._alpha_velocity: pd.Series | None = None

    def load(self, start: str = _DATA_START, end: str = _DATA_END_EXCLUSIVE) -> "BetaCalculator":
        """Load close prices for target and benchmark via HistoricalDataLoader."""
        loader = HistoricalDataLoader()
        target = loader.get_ohlcv(self.target_ticker, start, end)["close_price"].rename(self.target_ticker)
        benchmark = loader.get_ohlcv(self.benchmark_ticker, start, end)["close_price"].rename(self.benchmark_ticker)

        # Align on intersection — handles HK holidays where index and stock may differ
        common_dates = target.index.intersection(benchmark.index)
        self._ticker_prices = target.loc[common_dates]
        self._benchmark_prices = benchmark.loc[common_dates]
        print(f"Loaded {len(common_dates)} common trading days ({common_dates[0].date()} → {common_dates[-1].date()})")
        return self

    def compute_rolling_beta(self) -> pd.Series:
        """Compute 60-day rolling OLS (Ordinary Least Squares) beta using gs_quant.timeseries.econometrics.beta()."""
        if self._ticker_prices is None:
            raise RuntimeError("Call load() before compute_rolling_beta()")

        # Window ramp=window: no partial-window betas emitted (avoids extreme early values)
        self._rolling_beta = gsq_beta(
            self._ticker_prices,
            self._benchmark_prices,
            Window(self.window, self.window),
            prices=True,
        ).rename("rolling_beta")
        return self._rolling_beta

    def compute_residual_returns(self) -> pd.Series:
        """
        Compute idiosyncratic (residual) daily returns.

        R_residual = R_ticker - (rolling_beta × R_benchmark)
        """
        if self._rolling_beta is None:
            self.compute_rolling_beta()

        r_ticker = self._ticker_prices.pct_change().rename("r_ticker")
        r_benchmark = self._benchmark_prices.pct_change().rename("r_benchmark")

        # Align all three series on their common index
        aligned = pd.concat([r_ticker, r_benchmark, self._rolling_beta], axis=1).dropna()
        self._residual_returns = (
            aligned["r_ticker"] - aligned["rolling_beta"] * aligned["r_benchmark"]
        ).rename("residual_return")
        return self._residual_returns

    def summary(self) -> None:
        """Print summary statistics for the beta decomposition."""
        if self._residual_returns is None:
            self.compute_residual_returns()

        beta_valid = self._rolling_beta.dropna()
        resid = self._residual_returns
        r_benchmark = self._benchmark_prices.pct_change().dropna()

        # Full-period OLS for R²
        r_ticker_full = self._ticker_prices.pct_change().dropna()
        common = r_ticker_full.index.intersection(r_benchmark.index)
        r_t = r_ticker_full.loc[common]
        r_b = r_benchmark.loc[common]
        corr_matrix = np.corrcoef(r_t.values, r_b.values)
        r_squared_full = corr_matrix[0, 1] ** 2

        resid_benchmark_corr = resid.corr(r_benchmark.reindex(resid.index))

        print("\n" + "=" * 55)
        print(f"  Beta Engine Summary: {self.target_ticker} vs {self.benchmark_ticker}")
        print(f"  Rolling window: {self.window} days")
        print("=" * 55)
        print(f"  Mean rolling β:             {beta_valid.mean():.4f}")
        print(f"  β std dev:                  {beta_valid.std():.4f}")
        print(f"  β range:                    [{beta_valid.min():.3f}, {beta_valid.max():.3f}]")
        print(f"  Mean residual return/day:   {resid.mean() * 100:.4f}%")
        print(f"  Ann. residual vol:          {resid.std() * np.sqrt(252) * 100:.2f}%")
        print(f"  Full-period R²:             {r_squared_full:.4f}")
        print(f"  Residual–benchmark corr:    {resid_benchmark_corr:.4f}  (should be ≈0)")

        av = self.get_alpha_velocity().dropna()
        if not av.empty:
            current_av = av.iloc[-1]
            regime = "GAINING relative strength" if current_av > 0 else "LOSING relative strength"
            print(f"  Alpha Velocity (cur):       {current_av:+.4f}  → {regime}")
            pct_positive = (av > 0).mean() * 100
            print(f"  Alpha Velocity >0:          {pct_positive:.1f}% of the time")
        print("=" * 55 + "\n")

    def plot(self, output_path: Path | None = None) -> Path:
        """
        Save a 3-panel chart:
          Panel 1 — Normalized price: target vs benchmark (base=100)
          Panel 2 — Rolling 60-day beta
          Panel 3 — Cumulative residual return (pure alpha equity curve)
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self._residual_returns is None:
            self.compute_residual_returns()

        if output_path is None:
            from research.engines.output_manager import OutputManager
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OutputManager().artifact_path(f"beta_engine_{ts}.png")

        # Normalized prices (base=100 at first common date)
        norm_target = self._ticker_prices / self._ticker_prices.iloc[0] * 100
        norm_bench = self._benchmark_prices / self._benchmark_prices.iloc[0] * 100

        # Cumulative residual return
        cum_resid = (1 + self._residual_returns).cumprod() * 100 - 100  # in %

        av = self.get_alpha_velocity().dropna()

        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
        fig.suptitle(
            f"Factor-Neutral Alpha Engine: {self.target_ticker} vs {self.benchmark_ticker}\n"
            f"Rolling {self.window}-day OLS Beta Decomposition",
            fontsize=13,
            fontweight="bold",
        )

        # Panel 1 — Normalized prices
        ax1 = axes[0]
        ax1.plot(norm_target.index, norm_target.values, label=self.target_ticker, color="#1f77b4", linewidth=1.4)
        ax1.plot(norm_bench.index, norm_bench.values, label=self.benchmark_ticker, color="#ff7f0e", linewidth=1.4, alpha=0.8)
        ax1.axhline(100, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax1.set_ylabel("Normalized Price (base=100)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_title("Price Performance")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        # Panel 2 — Rolling beta
        ax2 = axes[1]
        beta_valid = self._rolling_beta.dropna()
        ax2.plot(beta_valid.index, beta_valid.values, color="#2ca02c", linewidth=1.4)
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, label="β=1 (market)")
        ax2.axhline(beta_valid.mean(), color="#d62728", linestyle=":", linewidth=1.0, label=f"mean β={beta_valid.mean():.2f}")
        ax2.fill_between(beta_valid.index, beta_valid.values, 1.0, alpha=0.15, color="#2ca02c")
        ax2.set_ylabel("Rolling Beta")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.set_title(f"Rolling {self.window}-Day Beta (1810.HK / ^HSTECH)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        # Panel 3 — Cumulative residual return
        ax3 = axes[2]
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in cum_resid.values]
        ax3.plot(cum_resid.index, cum_resid.values, color="#9467bd", linewidth=1.4)
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax3.fill_between(cum_resid.index, cum_resid.values, 0,
                         where=cum_resid.values >= 0, alpha=0.2, color="#1f77b4", label="Positive alpha")
        ax3.fill_between(cum_resid.index, cum_resid.values, 0,
                         where=cum_resid.values < 0, alpha=0.2, color="#d62728", label="Negative alpha")
        ax3.set_ylabel("Cumulative Residual Return (%)")
        ax3.legend(loc="upper left", fontsize=9)
        ax3.set_title("Cumulative Residual Return — Pure Idiosyncratic Alpha")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        # Panel 4 — Alpha Velocity (MA5 - MA20 on cumulative residual)
        ax4 = axes[3]
        ax4.plot(av.index, av.values, color="#8c564b", linewidth=1.4)
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax4.fill_between(av.index, av.values, 0,
                         where=av.values >= 0, alpha=0.25, color="#1f77b4", label="Gaining strength")
        ax4.fill_between(av.index, av.values, 0,
                         where=av.values < 0, alpha=0.25, color="#d62728", label="Losing strength")
        ax4.set_ylabel("Alpha Velocity")
        ax4.set_xlabel("Date")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.set_title("Alpha Velocity — MA(5) − MA(20) of Cumulative Residual Return")
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved → {output_path}")
        return output_path

    def compute_alpha_velocity(self, fast: int = 5, slow: int = 20) -> pd.Series:
        """
        Alpha Velocity = MA(fast) - MA(slow) applied to the cumulative residual return curve.

        Positive → alpha curve is in a short-term uptrend vs its longer-term average (gaining relative strength).
        Negative → alpha is decelerating or reversing.
        """
        if self._residual_returns is None:
            self.compute_residual_returns()

        cum_resid = (1 + self._residual_returns).cumprod() * 100 - 100
        ma_fast = gsq_ma(cum_resid, Window(fast, fast)).rename(f"ma{fast}")
        ma_slow = gsq_ma(cum_resid, Window(slow, slow)).rename(f"ma{slow}")
        self._alpha_velocity = (ma_fast - ma_slow).rename("alpha_velocity")
        return self._alpha_velocity

    def get_alpha_velocity(self) -> pd.Series:
        """Return the alpha velocity series (computed if needed)."""
        if self._alpha_velocity is None:
            self.compute_alpha_velocity()
        return self._alpha_velocity

    def get_residual_returns(self) -> pd.Series:
        """Return the residual return series (computed if needed). Used by downstream modules."""
        if self._residual_returns is None:
            self.compute_residual_returns()
        return self._residual_returns

    def get_rolling_beta(self) -> pd.Series:
        """Return the rolling beta series (computed if needed)."""
        if self._rolling_beta is None:
            self.compute_rolling_beta()
        return self._rolling_beta


def main() -> None:
    parser = argparse.ArgumentParser(description="M5.1 Beta Engine — 1810.HK vs ^HSTECH residual returns")
    parser.add_argument("--target", default="1810.HK", help="Target ticker (default: 1810.HK)")
    parser.add_argument("--benchmark", default="3033.HK", help="Benchmark ticker (default: 3033.HK = CSOP HSTECH ETF)")
    parser.add_argument("--window", type=int, default=60, help="Rolling OLS window in days (default: 60)")
    parser.add_argument("--start", default=_DATA_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=_DATA_END_EXCLUSIVE, help="End date YYYY-MM-DD (exclusive)")
    args = parser.parse_args()

    calc = BetaCalculator(target_ticker=args.target, benchmark_ticker=args.benchmark, window=args.window)
    calc.load(start=args.start, end=args.end)
    calc.compute_rolling_beta()
    calc.compute_residual_returns()
    calc.compute_alpha_velocity()
    calc.summary()
    calc.plot()


if __name__ == "__main__":
    main()
