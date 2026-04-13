"""
Anomaly Detectors — Isolation Forest & Ensemble
================================================
Provides IsolationForestDetector (wrapping sklearn's IsolationForest)
and EnsembleAnomalyDetector for combining multiple detectors via
normalized score averaging.

AnomalyDetector is kept as a backward-compatible alias for
IsolationForestDetector.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from typing import List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Wraps sklearn's IsolationForest for FX anomaly detection.

    Anomaly scores:
      - Prediction: -1 = anomaly, 1 = normal
      - Decision score: closer to -1 = more anomalous,
                        closer to +1 = more normal
      - Normalized score: 0 = normal, 1 = anomalous
    """

    name = "IsolationForest"

    def __init__(self, **kwargs):
        """
        Initialize the Isolation Forest model.

        Default parameters from config can be overridden via kwargs.
        """
        params = {**config.IFOREST_PARAMS, **kwargs}
        self.model = IsolationForest(**params)
        self.is_fitted = False
        self.feature_names = None
        logger.info(f"AnomalyDetector initialized with params: {params}")

    def train(self, features: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on the feature matrix.

        Args:
            features: DataFrame of engineered features (from AnomalyFeaturePipeline)

        Returns:
            self (for chaining)
        """
        logger.info(
            f"Training Isolation Forest on {features.shape[0]} samples "
            f"with {features.shape[1]} features..."
        )

        self.feature_names = list(features.columns)
        self.model.fit(features.values)
        self.is_fitted = True

        # Count anomalies in training data
        predictions = self.model.predict(features.values)
        n_anomalies = (predictions == -1).sum()
        anomaly_pct = n_anomalies / len(predictions) * 100

        logger.info(
            f"Training complete. Detected {n_anomalies} anomalies "
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
            raise RuntimeError("Model has not been trained. Call train() first.")

        labels = self.model.predict(features.values)
        scores = self.model.decision_function(features.values)

        # Normalize scores to [0, 1] where 1 = most anomalous
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
            f"Prediction complete: {n_anomalies} anomalies detected "
            f"out of {len(labels)} samples."
        )
        return results

    def get_anomaly_dates(self, features: pd.DataFrame,
                          fx_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get detailed information about detected anomalies.

        Args:
            features: Feature matrix used for prediction
            fx_data: Optional original FX data for context

        Returns:
            DataFrame of anomaly details sorted by severity.
        """
        results = self.predict(features)
        anomalies = results[results["anomaly_label"] == -1].copy()
        anomalies = anomalies.sort_values("anomaly_score_normalized",
                                           ascending=False)

        if fx_data is not None:
            # Add original FX rates for context
            common_idx = anomalies.index.intersection(fx_data.index)
            anomalies = anomalies.loc[common_idx]
            for col in fx_data.columns:
                if col in fx_data.columns:
                    anomalies[f"rate_{col}"] = fx_data.loc[common_idx, col]

        return anomalies

    def serialize(self, path: str = None):
        """Save the trained model to disk via joblib."""
        path = path or config.ANOMALY_MODEL_PATH
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an untrained model.")

        artifact = {
            "model": self.model,
            "feature_names": self.feature_names,
        }
        joblib.dump(artifact, path)
        logger.info(f"Model serialized to {path}")

    @classmethod
    def load(cls, path: str = None) -> "IsolationForestDetector":
        """Load a trained model from disk."""
        path = path or config.ANOMALY_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = joblib.load(path)
        detector = cls.__new__(cls)
        detector.model = artifact["model"]
        detector.feature_names = artifact["feature_names"]
        detector.is_fitted = True

        logger.info(f"Model loaded from {path}")
        return detector

    def get_feature_importance(self, features: pd.DataFrame,
                                n_top: int = 10) -> pd.DataFrame:
        """
        Estimate feature importance by permuting each feature and
        measuring the change in average anomaly score.

        Args:
            features: Feature matrix
            n_top: Number of top features to return

        Returns:
            DataFrame with feature importance scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained.")

        baseline_scores = self.model.decision_function(features.values)
        baseline_mean = baseline_scores.mean()

        importances = {}
        for i, col in enumerate(features.columns):
            # Permute this feature
            permuted = features.values.copy()
            np.random.shuffle(permuted[:, i])

            permuted_scores = self.model.decision_function(permuted)
            importance = abs(baseline_mean - permuted_scores.mean())
            importances[col] = importance

        importance_df = pd.DataFrame.from_dict(
            importances, orient="index", columns=["importance"]
        )
        importance_df = importance_df.sort_values(
            "importance", ascending=False
        ).head(n_top)

        return importance_df


# Backward-compatible alias
AnomalyDetector = IsolationForestDetector


class EnsembleAnomalyDetector:
    """
    Combines multiple anomaly detectors by averaging their normalized scores.

    Each detector must expose:
      - train(features) -> self
      - predict(features) -> DataFrame with anomaly_score_normalized column

    The ensemble threshold is set at the (1 - contamination) percentile
    of the averaged score distribution.
    """

    name = "Ensemble"

    def __init__(self, detectors: list, weights: Optional[List[float]] = None,
                 contamination: float = 0.01):
        """
        Args:
            detectors: List of initialized (but untrained) detector objects.
            weights: Optional per-detector weights (will be L1-normalized).
                     If None, equal weights are used.
            contamination: Anomaly fraction for thresholding.
        """
        if not detectors:
            raise ValueError("At least one detector is required.")

        self.detectors = detectors
        self.contamination = contamination
        self.is_fitted = False
        self.feature_names = None
        self.threshold = None

        if weights is None:
            self.weights = [1.0 / len(detectors)] * len(detectors)
        else:
            if len(weights) != len(detectors):
                raise ValueError("len(weights) must equal len(detectors).")
            total = sum(weights)
            self.weights = [w / total for w in weights]

        names = [
            getattr(d, "name", type(d).__name__) for d in detectors
        ]
        self.name = "Ensemble(" + "+".join(names) + ")"
        logger.info(f"EnsembleAnomalyDetector created: {self.name}")

    def train(self, features: pd.DataFrame) -> "EnsembleAnomalyDetector":
        """Train all constituent detectors on the feature matrix."""
        logger.info(f"Training ensemble: {self.name}")
        self.feature_names = list(features.columns)

        for det in self.detectors:
            det.train(features)

        # Calibrate ensemble threshold on training data
        results = self.predict(features)
        self.threshold = float(
            np.percentile(
                results["anomaly_score_normalized"].values,
                (1 - self.contamination) * 100,
            )
        )
        self.is_fitted = True
        logger.info(f"Ensemble trained. Score threshold = {self.threshold:.4f}")
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict by averaging weighted normalized scores across detectors."""
        # Check that constituent detectors are trained (allows internal call from train())
        unfitted = [
            getattr(d, "name", type(d).__name__)
            for d in self.detectors
            if not getattr(d, "is_fitted", False)
        ]
        if unfitted:
            raise RuntimeError(
                f"EnsembleAnomalyDetector: these sub-detectors are not fitted: {unfitted}. "
                "Call train() first."
            )

        score_matrix = np.zeros((len(features), len(self.detectors)))
        for i, (det, w) in enumerate(zip(self.detectors, self.weights)):
            pred = det.predict(features)
            score_matrix[:, i] = pred["anomaly_score_normalized"].values * w

        avg_scores = score_matrix.sum(axis=1)  # weighted sum (weights sum to 1)

        # Normalize averaged scores to [0, 1]
        s_min = avg_scores.min()
        s_max = avg_scores.max()
        s_range = s_max - s_min if s_max != s_min else 1.0
        normalized = (avg_scores - s_min) / s_range

        # Use calibrated threshold if available, otherwise use contamination percentile
        if self.threshold is not None:
            labels = np.where(normalized >= self.threshold, -1, 1)
        else:
            thr = np.percentile(normalized, (1 - self.contamination) * 100)
            labels = np.where(normalized >= thr, -1, 1)

        results = pd.DataFrame({
            "anomaly_label": labels,
            "anomaly_score": avg_scores,
            "anomaly_score_normalized": normalized,
        }, index=features.index)

        n_anomalies = (labels == -1).sum()
        logger.info(
            f"Ensemble prediction: {n_anomalies} anomalies out of {len(labels)} samples."
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
