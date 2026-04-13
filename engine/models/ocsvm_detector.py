"""
One-Class SVM Anomaly Detector
================================
Wraps sklearn's OneClassSVM for FX anomaly detection.

The One-Class SVM learns a decision boundary around the normal
training data in a kernel-induced feature space. Points outside
the boundary are flagged as anomalies.

Uses RBF kernel (default) and exposes the same interface as
IsolationForestDetector so it can be swapped in or combined
via EnsembleAnomalyDetector.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class OCSVMDetector:
    """
    One-Class SVM anomaly detector for FX market data.

    Anomaly scores:
      - Prediction: -1 = anomaly, 1 = normal
      - Decision score: closer to -1 = more anomalous
      - Normalized score: 0 = normal, 1 = anomalous
    """

    def __init__(self, **kwargs):
        params = {**config.OCSVM_PARAMS, **kwargs}
        self.model = OneClassSVM(**params)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        logger.info(f"OCSVMDetector initialized with params: {params}")

    def train(self, features: pd.DataFrame) -> "OCSVMDetector":
        """
        Fit the One-Class SVM on the feature matrix.

        OCSVM is sensitive to scale so features are standardised
        internally — the scaler is stored alongside the model.

        Args:
            features: DataFrame of engineered features

        Returns:
            self (for chaining)
        """
        logger.info(
            f"Training One-Class SVM on {features.shape[0]} samples "
            f"with {features.shape[1]} features..."
        )

        self.feature_names = list(features.columns)
        X = self.scaler.fit_transform(features.values)
        self.model.fit(X)
        self.is_fitted = True

        # Count anomalies in training data
        predictions = self.model.predict(X)
        n_anomalies = (predictions == -1).sum()
        anomaly_pct = n_anomalies / len(predictions) * 100

        logger.info(
            f"OCSVM training complete. Detected {n_anomalies} anomalies "
            f"({anomaly_pct:.2f}%) in training data."
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly labels and scores for new data.

        Args:
            features: DataFrame of engineered features

        Returns:
            DataFrame with columns:
              - anomaly_label: -1 (anomaly) or 1 (normal)
              - anomaly_score: raw decision function score
              - anomaly_score_normalized: 0 (normal) to 1 (anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("OCSVMDetector has not been trained. Call train() first.")

        X = self.scaler.transform(features.values)
        labels = self.model.predict(X)
        scores = self.model.decision_function(X)

        # Normalize scores: lower raw score = more anomalous → flip and scale to [0,1]
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min if score_max != score_min else 1.0
        normalized = 1 - (scores - score_min) / score_range

        results = pd.DataFrame({
            "anomaly_label": labels,
            "anomaly_score": scores,
            "anomaly_score_normalized": normalized,
        }, index=features.index)

        n_anomalies = (labels == -1).sum()
        logger.info(
            f"OCSVM prediction: {n_anomalies} anomalies detected "
            f"out of {len(labels)} samples."
        )
        return results

    def get_anomaly_dates(self, features: pd.DataFrame,
                          fx_data: pd.DataFrame = None) -> pd.DataFrame:
        """Get detailed anomaly information, sorted by severity."""
        results = self.predict(features)
        anomalies = results[results["anomaly_label"] == -1].copy()
        anomalies = anomalies.sort_values("anomaly_score_normalized", ascending=False)

        if fx_data is not None:
            common_idx = anomalies.index.intersection(fx_data.index)
            anomalies = anomalies.loc[common_idx]
            for col in fx_data.columns:
                anomalies[f"rate_{col}"] = fx_data.loc[common_idx, col]

        return anomalies

    def serialize(self, path: str = None):
        """Save the trained model and scaler to disk."""
        path = path or config.OCSVM_MODEL_PATH
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an untrained OCSVM model.")

        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        joblib.dump(artifact, path)
        logger.info(f"OCSVMDetector serialized to {path}")

    @classmethod
    def load(cls, path: str = None) -> "OCSVMDetector":
        """Load a trained model from disk."""
        path = path or config.OCSVM_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = joblib.load(path)
        detector = cls.__new__(cls)
        detector.model = artifact["model"]
        detector.scaler = artifact["scaler"]
        detector.feature_names = artifact["feature_names"]
        detector.is_fitted = True

        logger.info(f"OCSVMDetector loaded from {path}")
        return detector
