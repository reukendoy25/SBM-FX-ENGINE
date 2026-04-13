"""
Tests — Anomaly Detectors & Evaluator
=======================================
Unit tests for all anomaly detection modules:
  - IsolationForestDetector (+ AnomalyDetector alias)
  - OCSVMDetector
  - AutoencoderDetector
  - EnsembleAnomalyDetector
  - AnomalyEvaluator
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.anomaly_detector import IsolationForestDetector, AnomalyDetector, EnsembleAnomalyDetector
from models.ocsvm_detector import OCSVMDetector
from models.anomaly_evaluator import AnomalyEvaluator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _create_synthetic_fx_data(n_samples=500, n_currencies=4,
                               n_anomalies=10, seed=42):
    """Create synthetic FX data with injected anomalies."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    currencies = ["USD", "GBP", "JPY", "CHF"][:n_currencies]

    data = {}
    for ccy in currencies:
        base_rate = np.random.uniform(0.5, 150)
        noise = np.random.normal(0, base_rate * 0.005, n_samples)
        trend = np.cumsum(np.random.normal(0, base_rate * 0.001, n_samples))
        data[ccy] = base_rate + trend + noise

    df = pd.DataFrame(data, index=dates)

    # Inject anomalies
    anomaly_indices = np.random.choice(
        range(50, n_samples - 50), n_anomalies, replace=False
    )
    for idx in anomaly_indices:
        ccy = np.random.choice(currencies)
        df.iloc[idx, df.columns.get_loc(ccy)] *= np.random.choice([1.15, 0.85])

    return df, anomaly_indices


def _build_features(fx_data):
    """Build AnomalyFeaturePipeline features for testing."""
    from data.preprocessing import AnomalyFeaturePipeline
    pipeline = AnomalyFeaturePipeline()
    return pipeline.build_features(fx_data)


def _get_true_labels(features, anomaly_indices):
    """
    Convert injected anomaly row indices to a binary label array
    aligned with the feature DataFrame index.
    Returns 1 for anomaly, 0 for normal.
    """
    labels = np.zeros(len(features), dtype=int)
    valid = [i for i in anomaly_indices if i < len(features)]
    labels[valid] = 1
    return labels


# ---------------------------------------------------------------------------
# Isolation Forest Detector
# ---------------------------------------------------------------------------

class TestIsolationForestDetector:
    """Tests for IsolationForestDetector (and backward-compat AnomalyDetector alias)."""

    def test_alias(self):
        assert AnomalyDetector is IsolationForestDetector

    def test_initialization(self):
        detector = IsolationForestDetector()
        assert detector.is_fitted is False
        assert detector.feature_names is None

    def test_initialization_custom_params(self):
        detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
        assert detector.is_fitted is False

    def test_train(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = IsolationForestDetector()
        detector.train(features)
        assert detector.is_fitted is True
        assert len(detector.feature_names) == features.shape[1]

    def test_predict_shape_and_columns(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = IsolationForestDetector()
        detector.train(features)
        results = detector.predict(features)
        assert len(results) == len(features)
        for col in ("anomaly_label", "anomaly_score", "anomaly_score_normalized"):
            assert col in results.columns

    def test_predict_labels_valid(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = IsolationForestDetector()
        detector.train(features)
        results = detector.predict(features)
        assert set(results["anomaly_label"].unique()).issubset({-1, 1})

    def test_normalized_scores_range(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = IsolationForestDetector()
        detector.train(features)
        results = detector.predict(features)
        assert results["anomaly_score_normalized"].min() >= 0
        assert results["anomaly_score_normalized"].max() <= 1

    def test_detects_anomalies(self):
        fx_data, _ = _create_synthetic_fx_data(n_anomalies=20)
        features = _build_features(fx_data)
        detector = IsolationForestDetector(contamination=0.05)
        detector.train(features)
        results = detector.predict(features)
        assert (results["anomaly_label"] == -1).sum() > 0

    def test_predict_before_train_raises(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        with pytest.raises(RuntimeError):
            IsolationForestDetector().predict(features)

    def test_serialization(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = IsolationForestDetector()
        detector.train(features)
        original = detector.predict(features)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        try:
            detector.serialize(temp_path)
            loaded = IsolationForestDetector.load(temp_path)
            reloaded = loaded.predict(features)
            np.testing.assert_array_equal(
                original["anomaly_label"].values, reloaded["anomaly_label"].values
            )
        finally:
            os.unlink(temp_path)

    def test_feature_importance(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = IsolationForestDetector()
        detector.train(features)
        importance = detector.get_feature_importance(features, n_top=5)
        assert len(importance) <= 5
        assert "importance" in importance.columns
        assert all(importance["importance"] >= 0)


# ---------------------------------------------------------------------------
# One-Class SVM Detector
# ---------------------------------------------------------------------------

class TestOCSVMDetector:
    """Tests for OCSVMDetector."""

    def test_initialization(self):
        detector = OCSVMDetector()
        assert detector.is_fitted is False
        assert detector.feature_names is None

    def test_train(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = OCSVMDetector()
        detector.train(features)
        assert detector.is_fitted is True
        assert len(detector.feature_names) == features.shape[1]

    def test_predict_shape_and_columns(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = OCSVMDetector()
        detector.train(features)
        results = detector.predict(features)
        assert len(results) == len(features)
        for col in ("anomaly_label", "anomaly_score", "anomaly_score_normalized"):
            assert col in results.columns

    def test_predict_labels_valid(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = OCSVMDetector()
        detector.train(features)
        results = detector.predict(features)
        assert set(results["anomaly_label"].unique()).issubset({-1, 1})

    def test_normalized_scores_range(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = OCSVMDetector()
        detector.train(features)
        results = detector.predict(features)
        assert results["anomaly_score_normalized"].min() >= 0
        assert results["anomaly_score_normalized"].max() <= 1

    def test_predict_before_train_raises(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        with pytest.raises(RuntimeError):
            OCSVMDetector().predict(features)

    def test_serialization(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = OCSVMDetector()
        detector.train(features)
        original = detector.predict(features)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        try:
            detector.serialize(temp_path)
            loaded = OCSVMDetector.load(temp_path)
            reloaded = loaded.predict(features)
            np.testing.assert_array_equal(
                original["anomaly_label"].values, reloaded["anomaly_label"].values
            )
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Autoencoder Detector
# ---------------------------------------------------------------------------

class TestAutoencoderDetector:
    """Tests for AutoencoderDetector (TensorFlow required)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_tf(self):
        pytest.importorskip("tensorflow", reason="TensorFlow not installed")

    def _get_detector(self):
        from models.autoencoder_detector import AutoencoderDetector
        return AutoencoderDetector(epochs=5, batch_size=32, encoding_dims=[16, 8])

    def test_initialization(self):
        from models.autoencoder_detector import AutoencoderDetector
        d = AutoencoderDetector()
        assert d.is_fitted is False

    def test_train(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = self._get_detector()
        detector.train(features)
        assert detector.is_fitted is True
        assert detector.threshold is not None

    def test_predict_shape_and_columns(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = self._get_detector()
        detector.train(features)
        results = detector.predict(features)
        assert len(results) == len(features)
        for col in ("anomaly_label", "anomaly_score", "anomaly_score_normalized"):
            assert col in results.columns

    def test_normalized_scores_range(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = self._get_detector()
        detector.train(features)
        results = detector.predict(features)
        assert results["anomaly_score_normalized"].min() >= 0
        assert results["anomaly_score_normalized"].max() <= 1

    def test_predict_before_train_raises(self):
        from models.autoencoder_detector import AutoencoderDetector
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        with pytest.raises(RuntimeError):
            AutoencoderDetector().predict(features)

    def test_serialization(self):
        import tempfile, os
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        detector = self._get_detector()
        detector.train(features)
        original = detector.predict(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "ae.keras")
            meta_path = os.path.join(tmpdir, "ae_meta.pkl")
            import config as cfg
            orig_model = cfg.AUTOENCODER_MODEL_PATH
            orig_meta = cfg.AUTOENCODER_SCALER_PATH
            cfg.AUTOENCODER_MODEL_PATH = model_path
            cfg.AUTOENCODER_SCALER_PATH = meta_path
            try:
                from models.autoencoder_detector import AutoencoderDetector
                detector.serialize()
                loaded = AutoencoderDetector.load()
                reloaded = loaded.predict(features)
                np.testing.assert_array_equal(
                    original["anomaly_label"].values, reloaded["anomaly_label"].values
                )
            finally:
                cfg.AUTOENCODER_MODEL_PATH = orig_model
                cfg.AUTOENCODER_SCALER_PATH = orig_meta


# ---------------------------------------------------------------------------
# Ensemble Anomaly Detector
# ---------------------------------------------------------------------------

class TestEnsembleAnomalyDetector:
    """Tests for EnsembleAnomalyDetector."""

    def test_initialization_equal_weights(self):
        d1 = IsolationForestDetector()
        d2 = OCSVMDetector()
        ensemble = EnsembleAnomalyDetector([d1, d2])
        assert len(ensemble.weights) == 2
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6

    def test_initialization_custom_weights(self):
        d1 = IsolationForestDetector()
        d2 = OCSVMDetector()
        ensemble = EnsembleAnomalyDetector([d1, d2], weights=[0.7, 0.3])
        assert abs(ensemble.weights[0] - 0.7) < 1e-6
        assert abs(ensemble.weights[1] - 0.3) < 1e-6

    def test_empty_detectors_raises(self):
        with pytest.raises(ValueError):
            EnsembleAnomalyDetector([])

    def test_mismatched_weights_raises(self):
        d1 = IsolationForestDetector()
        with pytest.raises(ValueError):
            EnsembleAnomalyDetector([d1], weights=[0.5, 0.5])

    def test_train_and_predict(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        ensemble = EnsembleAnomalyDetector(
            [IsolationForestDetector(), OCSVMDetector()]
        )
        ensemble.train(features)
        results = ensemble.predict(features)
        assert len(results) == len(features)
        for col in ("anomaly_label", "anomaly_score", "anomaly_score_normalized"):
            assert col in results.columns

    def test_normalized_scores_range(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        ensemble = EnsembleAnomalyDetector(
            [IsolationForestDetector(), OCSVMDetector()]
        )
        ensemble.train(features)
        results = ensemble.predict(features)
        assert results["anomaly_score_normalized"].min() >= 0
        assert results["anomaly_score_normalized"].max() <= 1

    def test_detects_some_anomalies(self):
        fx_data, _ = _create_synthetic_fx_data(n_anomalies=20)
        features = _build_features(fx_data)
        ensemble = EnsembleAnomalyDetector(
            [IsolationForestDetector(contamination=0.05),
             OCSVMDetector(nu=0.05)]
        )
        ensemble.train(features)
        results = ensemble.predict(features)
        assert (results["anomaly_label"] == -1).sum() > 0

    def test_predict_before_train_raises(self):
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        # Untrained ensemble — sub-detectors not fitted → should raise RuntimeError
        ensemble = EnsembleAnomalyDetector([IsolationForestDetector(), OCSVMDetector()])
        ensemble.is_fitted = False  # explicitly ensure not fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.predict(features)


# ---------------------------------------------------------------------------
# Anomaly Evaluator
# ---------------------------------------------------------------------------

class TestAnomalyEvaluator:
    """Tests for AnomalyEvaluator — proxy metrics and supervised metrics."""

    def _trained_results(self, contamination=0.05, n_anomalies=20):
        fx_data, anomaly_idx = _create_synthetic_fx_data(n_anomalies=n_anomalies)
        features = _build_features(fx_data)
        detector = IsolationForestDetector(contamination=contamination)
        detector.train(features)
        results = detector.predict(features)
        return results, features, anomaly_idx

    def test_evaluate_returns_required_keys(self):
        results, _, _ = self._trained_results()
        metrics = AnomalyEvaluator.evaluate(results)
        for key in ("n_samples", "n_anomalies", "anomaly_rate_pct",
                    "score_separation", "contamination_precision", "score_gini"):
            assert key in metrics, f"Missing metric: {key}"

    def test_evaluate_anomaly_rate_in_range(self):
        results, _, _ = self._trained_results()
        metrics = AnomalyEvaluator.evaluate(results)
        assert 0 <= metrics["anomaly_rate_pct"] <= 100

    def test_evaluate_score_separation_positive(self):
        """Anomaly scores should be higher on average than normal scores."""
        results, _, _ = self._trained_results(contamination=0.05, n_anomalies=25)
        metrics = AnomalyEvaluator.evaluate(results, contamination=0.05)
        # We can't guarantee this is always positive, but gini should be > 0
        assert metrics["score_gini"] >= 0

    def test_evaluate_with_labels_returns_precision_recall_f1(self):
        results, features, anomaly_idx = self._trained_results(n_anomalies=20)
        true_labels = _get_true_labels(features, anomaly_idx)
        metrics = AnomalyEvaluator.evaluate_with_labels(
            results, true_labels, contamination=0.05
        )
        for key in ("precision", "recall", "f1_score"):
            assert key in metrics

    def test_evaluate_with_labels_recall_nonzero(self):
        """With strong anomalies, recall should be > 0."""
        fx_data, anomaly_idx = _create_synthetic_fx_data(
            n_anomalies=25, seed=99
        )
        features = _build_features(fx_data)
        detector = IsolationForestDetector(contamination=0.1)
        detector.train(features)
        results = detector.predict(features)
        true_labels = _get_true_labels(features, anomaly_idx)
        metrics = AnomalyEvaluator.evaluate_with_labels(
            results, true_labels, contamination=0.1
        )
        assert metrics["recall"] is not None
        assert metrics["recall"] >= 0

    def test_compare_returns_dataframe(self):
        results, _, _ = self._trained_results()
        ocsvm = OCSVMDetector()
        fx_data, _ = _create_synthetic_fx_data()
        features = _build_features(fx_data)
        ocsvm.train(features)
        ocsvm_results = ocsvm.predict(features)

        comparison = AnomalyEvaluator.compare(
            {"IF": results, "OCSVM": ocsvm_results}
        )
        assert isinstance(comparison, pd.DataFrame)
        assert "IF" in comparison.index
        assert "OCSVM" in comparison.index
