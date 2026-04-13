"""
ECB Data Fetcher
================
Connects to the European Central Bank Statistical Data Warehouse REST API
to extract historical FX daily rates for specified currency pairs against EUR.

API Reference: https://data-api.ecb.europa.eu/
"""

import logging
import pandas as pd
import requests
from io import StringIO

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class ECBFetcher:
    """Fetches historical FX rates from the ECB Statistical Data Warehouse."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.ECB_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "text/csv",
            "Accept-Encoding": "gzip, deflate",
        })

    def fetch(self, currency: str, frequency: str = "D",
              start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch daily FX rate for a single currency against EUR.

        Args:
            currency: ISO 4217 currency code (e.g., 'USD', 'GBP')
            frequency: Data frequency ('D' for daily, 'M' for monthly)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with DatetimeIndex and columns [rate, currency]
        """
        start_date = start_date or config.DATA_START_DATE
        end_date = end_date or config.DATA_END_DATE

        # ECB SDMX REST API key structure: D.{CCY}.EUR.SP00.A
        flow_ref = "EXR"
        key = f"{frequency}.{currency}.EUR.SP00.A"
        url = f"{self.base_url}/{key}"

        params = {
            "startPeriod": start_date,
            "endPeriod": end_date,
            "format": config.ECB_API_FORMAT,
        }

        logger.info(f"Fetching {currency}/EUR from ECB API: {url}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"ECB API request failed for {currency}: {e}")
            raise ConnectionError(
                f"Failed to fetch {currency}/EUR data from ECB: {e}"
            ) from e

        # Parse CSV response
        df = self._parse_csv_response(response.text, currency)
        logger.info(
            f"Fetched {len(df)} records for {currency}/EUR "
            f"({df.index.min().date()} to {df.index.max().date()})"
        )
        return df

    def fetch_multiple(self, currencies: list = None,
                       start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """
        Fetch FX rates for multiple currencies and return a combined DataFrame.

        Returns:
            DataFrame with DatetimeIndex and one column per currency.
        """
        currencies = currencies or config.CURRENCY_PAIRS
        frames = {}

        for ccy in currencies:
            try:
                df = self.fetch(ccy, start_date=start_date, end_date=end_date)
                frames[ccy] = df["rate"]
            except ConnectionError as e:
                logger.warning(f"Skipping {ccy}: {e}")
                continue

        if not frames:
            raise RuntimeError("Failed to fetch data for any currency pair.")

        combined = pd.DataFrame(frames)
        combined.index.name = "date"
        combined = combined.sort_index()

        # Forward-fill missing values (weekends/holidays already excluded by ECB)
        combined = combined.ffill()

        logger.info(
            f"Combined dataset: {combined.shape[0]} rows × "
            f"{combined.shape[1]} currencies"
        )
        return combined

    def _parse_csv_response(self, csv_text: str,
                            currency: str) -> pd.DataFrame:
        """Parse ECB CSV response into a clean DataFrame."""
        try:
            df = pd.read_csv(StringIO(csv_text))
        except Exception as e:
            raise ValueError(f"Failed to parse ECB CSV response: {e}") from e

        # ECB CSV columns include TIME_PERIOD and OBS_VALUE
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            raise ValueError(
                f"Unexpected CSV format from ECB. "
                f"Columns found: {list(df.columns)}"
            )

        result = pd.DataFrame({
            "date": pd.to_datetime(df["TIME_PERIOD"]),
            "rate": pd.to_numeric(df["OBS_VALUE"], errors="coerce"),
            "currency": currency,
        })

        result = result.dropna(subset=["rate"])
        result = result.set_index("date").sort_index()

        return result

    def save_to_cache(self, df: pd.DataFrame, filename: str = "ecb_fx_data.csv"):
        """Save fetched data to local cache."""
        path = os.path.join(config.DATA_DIR, filename)
        df.to_csv(path)
        logger.info(f"Data cached to {path}")

    def load_from_cache(self, filename: str = "ecb_fx_data.csv") -> pd.DataFrame:
        """Load data from local cache."""
        path = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache file not found: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} records from cache")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = ECBFetcher()

    # Fetch all configured currency pairs
    try:
        df = fetcher.fetch_multiple()
        fetcher.save_to_cache(df)
        print(f"\n{'='*60}")
        print(f"ECB FX Data Summary")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
        print(f"\nSample data (last 5 rows):")
        print(df.tail())
    except Exception as e:
        print(f"Error: {e}")
