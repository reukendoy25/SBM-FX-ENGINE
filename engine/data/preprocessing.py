"""
Data Preprocessing & Feature Engineering
=========================================
Transforms raw FX data and macro indicators into ML-ready features
for both the Isolation Forest anomaly detector and the LSTM forecaster.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers technical and statistical features from raw FX time series
    for anomaly detection and forecasting models.
    """

    @staticmethod
    def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns for all columns."""
        returns = np.log(df / df.shift(1))
        returns.columns = [f"{col}_log_return" for col in df.columns]
        return returns.dropna()

    @staticmethod
    def compute_rolling_volatility(df: pd.DataFrame,
                                    windows: list = None) -> pd.DataFrame:
        """
        Compute rolling standard deviation (volatility) of log returns.

        Args:
            df: DataFrame of raw FX rates
            windows: List of rolling window sizes (in days)
        """
        windows = windows or config.ROLLING_WINDOWS
        log_returns = np.log(df / df.shift(1))
        frames = []

        for window in windows:
            vol = log_returns.rolling(window=window).std() * np.sqrt(252)
            vol.columns = [f"{col}_vol_{window}d" for col in df.columns]
            frames.append(vol)

        return pd.concat(frames, axis=1).dropna()

    @staticmethod
    def compute_momentum(df: pd.DataFrame,
                          window: int = None) -> pd.DataFrame:
        """
        Compute RSI-style momentum indicator.

        RSI = 100 - (100 / (1 + RS))
        where RS = avg_gain / avg_loss over the window period
        """
        window = window or config.MOMENTUM_WINDOW
        delta = df.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Avoid division by zero
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi.columns = [f"{col}_rsi_{window}" for col in df.columns]

        return rsi.dropna()

    @staticmethod
    def compute_bollinger_width(df: pd.DataFrame,
                                 window: int = 21) -> pd.DataFrame:
        """
        Compute Bollinger Band width as a volatility measure.
        Width = (Upper Band - Lower Band) / Middle Band
        """
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()

        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std

        width = (upper - lower) / (rolling_mean + 1e-10)
        width.columns = [f"{col}_bb_width_{window}" for col in df.columns]

        return width.dropna()

    @staticmethod
    def compute_rate_of_change(df: pd.DataFrame,
                                periods: list = None) -> pd.DataFrame:
        """Compute rate of change over multiple periods."""
        periods = periods or [5, 10, 21]
        frames = []

        for period in periods:
            roc = df.pct_change(periods=period)
            roc.columns = [f"{col}_roc_{period}d" for col in df.columns]
            frames.append(roc)

        return pd.concat(frames, axis=1).dropna()

    @staticmethod
    def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute spreads between key currency pairs.
        Useful for detecting correlated movements / divergences.
        """
        spreads = pd.DataFrame(index=df.index)

        pairs = [
            ("USD", "GBP"),
            ("USD", "JPY"),
            ("GBP", "CHF"),
            ("AUD", "USD"),
        ]

        for ccy1, ccy2 in pairs:
            if ccy1 in df.columns and ccy2 in df.columns:
                spreads[f"{ccy1}_{ccy2}_spread"] = (
                    df[ccy1] / (df[ccy2] + 1e-10)
                )

        return spreads.dropna() if not spreads.empty else spreads


class AnomalyFeaturePipeline:
    """
    Constructs the full feature matrix for Isolation Forest anomaly detection.
    """

    def __init__(self):
        self.engineer = FeatureEngineer()

    def build_features(self, fx_data: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive feature set for anomaly detection.

        Args:
            fx_data: DataFrame with DatetimeIndex and currency columns.

        Returns:
            Feature matrix ready for Isolation Forest training.
        """
        logger.info("Building anomaly detection features...")

        feature_frames = [
            self.engineer.compute_log_returns(fx_data),
            self.engineer.compute_rolling_volatility(fx_data),
            self.engineer.compute_momentum(fx_data),
            self.engineer.compute_bollinger_width(fx_data),
            self.engineer.compute_rate_of_change(fx_data),
            self.engineer.compute_spread(fx_data),
        ]

        # Align all features on the same index
        features = pd.concat(feature_frames, axis=1)
        features = features.dropna()

        # Remove any infinite values
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(
            f"Anomaly feature matrix: {features.shape[0]} rows × "
            f"{features.shape[1]} features"
        )
        return features


class LSTMFeaturePipeline:
    """
    Constructs the feature matrix and sequences for LSTM forecasting.
    """

    def __init__(self, sequence_length: int = None):
        self.sequence_length = (
            sequence_length or config.LSTM_PARAMS["sequence_length"]
        )
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.engineer = FeatureEngineer()

    def build_features(self, fx_data: pd.DataFrame,
                       macro_data: pd.DataFrame = None,
                       target_currency: str = "USD") -> pd.DataFrame:
        """
        Build feature set for LSTM forecasting.

        Args:
            fx_data: Raw FX rates DataFrame
            macro_data: Optional macro indicators DataFrame
            target_currency: Currency to forecast

        Returns:
            Combined feature DataFrame (unscaled)
        """
        logger.info(f"Building LSTM features for {target_currency}...")

        # Start with raw rates
        features = fx_data.copy()

        # Add log returns
        returns = self.engineer.compute_log_returns(fx_data)
        features = features.join(returns, how="inner")

        # Add volatility
        vol = self.engineer.compute_rolling_volatility(
            fx_data, windows=[5, 21]
        )
        features = features.join(vol, how="inner")

        # Add momentum
        rsi = self.engineer.compute_momentum(fx_data)
        features = features.join(rsi, how="inner")

        # Add macro indicators if available
        if macro_data is not None and not macro_data.empty:
            features = features.join(macro_data, how="inner")

        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(
            f"LSTM feature matrix: {features.shape[0]} rows × "
            f"{features.shape[1]} features"
        )
        return features

    def create_sequences(self, features: pd.DataFrame,
                          target_col: str) -> tuple:
        """
        Create input/output sequences for LSTM training.

        Args:
            features: Feature DataFrame
            target_col: Column name of the target variable

        Returns:
            (X, y, dates) where X has shape (samples, seq_len, n_features),
            y has shape (samples,), dates contains the target dates.
        """
        if target_col not in features.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available: {list(features.columns)}"
            )

        # Scale features
        data_scaled = self.scaler.fit_transform(features.values)
        target_idx = list(features.columns).index(target_col)

        X, y, dates = [], [], []

        for i in range(self.sequence_length, len(data_scaled)):
            X.append(data_scaled[i - self.sequence_length: i])
            y.append(data_scaled[i, target_idx])
            dates.append(features.index[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(
            f"Created {len(X)} sequences of length {self.sequence_length} "
            f"with {X.shape[2]} features"
        )
        return X, y, dates

    def split_train_test(self, X: np.ndarray, y: np.ndarray,
                          dates: list,
                          split_ratio: float = None) -> dict:
        """
        Split data into train and test sets chronologically.

        Returns:
            Dict with keys: X_train, X_test, y_train, y_test,
                           dates_train, dates_test
        """
        split_ratio = split_ratio or config.TRAIN_TEST_SPLIT
        split_idx = int(len(X) * split_ratio)

        return {
            "X_train": X[:split_idx],
            "X_test": X[split_idx:],
            "y_train": y[:split_idx],
            "y_test": y[split_idx:],
            "dates_train": dates[:split_idx],
            "dates_test": dates[split_idx:],
        }

    def inverse_transform_predictions(self, predictions: np.ndarray,
                                       n_features: int,
                                       target_idx: int) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.

        Args:
            predictions: Scaled prediction values
            n_features: Total number of features used in scaling
            target_idx: Index of target feature

        Returns:
            Predictions in original scale
        """
        # Create a dummy array of zeros with the same number of features
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, target_idx] = predictions.flatten()

        inversed = self.scaler.inverse_transform(dummy)
        return inversed[:, target_idx]
