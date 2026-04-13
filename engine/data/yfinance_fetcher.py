"""
yfinance Data Fetcher
=====================
Downloads supplementary macro indicators and FX pairs from Yahoo Finance
as fallback/supplement to the ECB API data.
"""

import logging
import pandas as pd
import yfinance as yf

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Fetches macro indicators and FX rates from Yahoo Finance."""

    def fetch_macro_indicators(self, start_date: str = None,
                                end_date: str = None) -> pd.DataFrame:
        """
        Fetch supplementary macro indicators:
        - 10Y Treasury Yield (^TNX)
        - VIX Volatility Index (^VIX)
        - S&P 500 (^GSPC)
        - Gold Futures (GC=F)
        - Oil Futures (CL=F)

        Returns:
            DataFrame with DatetimeIndex and one column per indicator.
        """
        start_date = start_date or config.DATA_START_DATE
        end_date = end_date or config.DATA_END_DATE

        frames = {}
        for name, ticker in config.YFINANCE_TICKERS.items():
            try:
                logger.info(f"Fetching {name} ({ticker}) from yfinance")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if not data.empty:
                    # Use closing price
                    if isinstance(data.columns, pd.MultiIndex):
                        frames[name] = data[("Close", ticker)]
                    else:
                        frames[name] = data["Close"]
                else:
                    logger.warning(f"No data returned for {name} ({ticker})")
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")

        if not frames:
            raise RuntimeError("Failed to fetch any macro indicators.")

        combined = pd.DataFrame(frames)
        combined.index.name = "date"
        combined = combined.sort_index().ffill()

        logger.info(
            f"Macro indicators: {combined.shape[0]} rows × "
            f"{combined.shape[1]} indicators"
        )
        return combined

    def fetch_fx_pairs(self, start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """
        Fetch FX pairs from Yahoo Finance as fallback to ECB API.

        Returns:
            DataFrame with DatetimeIndex and one column per pair.
        """
        start_date = start_date or config.DATA_START_DATE
        end_date = end_date or config.DATA_END_DATE

        frames = {}
        for name, ticker in config.YFINANCE_FX_PAIRS.items():
            try:
                logger.info(f"Fetching {name} ({ticker}) from yfinance")
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        frames[name] = data[("Close", ticker)]
                    else:
                        frames[name] = data["Close"]
                else:
                    logger.warning(f"No data returned for {name}")
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")

        if not frames:
            raise RuntimeError("Failed to fetch any FX pairs from yfinance.")

        combined = pd.DataFrame(frames)
        combined.index.name = "date"
        combined = combined.sort_index().ffill()

        logger.info(
            f"yfinance FX data: {combined.shape[0]} rows × "
            f"{combined.shape[1]} pairs"
        )
        return combined

    def save_to_cache(self, df: pd.DataFrame, filename: str):
        """Save fetched data to local cache."""
        path = os.path.join(config.DATA_DIR, filename)
        df.to_csv(path)
        logger.info(f"Data cached to {path}")

    def load_from_cache(self, filename: str) -> pd.DataFrame:
        """Load data from local cache."""
        path = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache file not found: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} records from cache")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = YFinanceFetcher()

    print("Fetching macro indicators...")
    try:
        macro_df = fetcher.fetch_macro_indicators()
        fetcher.save_to_cache(macro_df, "macro_indicators.csv")
        print(f"\nMacro Indicators Shape: {macro_df.shape}")
        print(macro_df.tail())
    except Exception as e:
        print(f"Error fetching macro indicators: {e}")

    print("\nFetching FX pairs...")
    try:
        fx_df = fetcher.fetch_fx_pairs()
        fetcher.save_to_cache(fx_df, "yfinance_fx_data.csv")
        print(f"\nFX Pairs Shape: {fx_df.shape}")
        print(fx_df.tail())
    except Exception as e:
        print(f"Error fetching FX pairs: {e}")
