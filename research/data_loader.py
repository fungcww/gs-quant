"""
Centralized historical market data loader.

Fetches OHLCV data exclusively from the internal API server.

Environment variables (loaded from .env at repo root):
    API_SERVER_URL  — base URL of the market data API server
                      e.g. http://192.168.31.208:8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_logger = logging.getLogger(__name__)

_DEFAULT_API_URL = os.getenv("API_SERVER_URL", "http://192.168.31.208:8000")
_REQUEST_TIMEOUT = 60  # seconds — first call may trigger a yfinance backfill on the server


class HistoricalDataLoader:
    """
    Load OHLCV data for a ticker over a date range from the internal API server.

    Endpoint: GET {api_url}/api/v1/market-data?ticker=...&start_date=...&end_date=...

    Returned DataFrame columns: open_price, high_price, low_price, close_price, volume
    Index: DatetimeIndex (daily, UTC-normalised), named 'date'
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_url = (api_url or _DEFAULT_API_URL).rstrip("/")
        self._headers = {"X-API-Key": api_key} if api_key else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Return a DataFrame of daily OHLCV bars for *ticker* in [start_date, end_date).

        Columns: open_price, high_price, low_price, close_price, volume
        Index:   DatetimeIndex named 'date'

        Raises RuntimeError if the API is unreachable or returns no data.
        """
        endpoint = f"{self._api_url}/api/v1/market-data"
        try:
            resp = requests.get(
                endpoint,
                params={"ticker": ticker, "start_date": start_date, "end_date": end_date},
                headers=self._headers,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"[HistoricalDataLoader] Cannot reach API server at {self._api_url}. "
                "Check that the server is running and API_SERVER_URL is set correctly in .env."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"[HistoricalDataLoader] API request timed out after {_REQUEST_TIMEOUT}s "
                f"(ticker={ticker})."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"[HistoricalDataLoader] API returned HTTP {exc.response.status_code} "
                f"for ticker={ticker}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"[HistoricalDataLoader] Unexpected error fetching ticker={ticker}: {exc}"
            ) from exc

        data = body.get("data", {})
        if not data or not data.get("time"):
            raise RuntimeError(
                f"[HistoricalDataLoader] API returned empty data for ticker={ticker!r} "
                f"between {start_date} and {end_date}."
            )

        _logger.info(
            "[HistoricalDataLoader] %s: %d bars (cache_status=%s)",
            ticker, body.get("count", 0), body.get("cache_status", "?"),
        )
        return self._normalise(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(data: dict) -> pd.DataFrame:
        """Build canonical DataFrame from the API's columnar response payload."""
        df = pd.DataFrame({
            "date":        data["time"],
            "open_price":  data["open"],
            "high_price":  data["high"],
            "low_price":   data["low"],
            "close_price": data.get("adj_close") or data["close"],
            "volume":      data["volume"],
        })
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()

        for col in ("open_price", "high_price", "low_price", "close_price"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

        close = df["close_price"]
        df["high_price"] = df["high_price"].fillna(close)
        df["low_price"] = df["low_price"].fillna(close)
        df["open_price"] = df["open_price"].fillna(close)

        return df[["open_price", "high_price", "low_price", "close_price", "volume"]]
