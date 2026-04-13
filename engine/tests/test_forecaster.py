"""
Tests — LSTM Forecaster
========================
Unit tests for the LSTM rate forecasting module.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _create_synthetic_sequences(n_samples=200, seq_len=30, n_features=5,
                                 seed=42):
    """Create synthetic sequence data for LSTM testing."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    return X, y


class TestLSTMForecaster:
    """Tests for the LSTMForecaster class."""

    def test_initialization(self):
        """Test model initialization."""
        from models.lstm_forecaster import LSTMForecaster
        forecaster = LSTMForecaster(epochs=1)
        assert forecaster.is_fitted is False
        assert forecaster.model is None

    def test_train_produces_model(self):
        """Test that training creates a model."""
        from models.lstm_forecaster import LSTMForecaster
        X_train, y_train = _create_synthetic_sequences(n_samples=50, seq_len=10)

        forecaster = LSTMForecaster(epochs=2, batch_size=16,
                                     early_stopping_patience=1)
        forecaster.train(X_train, y_train)

        assert forecaster.is_fitted is True
        assert forecaster.model is not None

    def test_predict_shape(self):
        """Test prediction output shape."""
        from models.lstm_forecaster import LSTMForecaster
        X_train, y_train = _create_synthetic_sequences(n_samples=50, seq_len=10)
        X_test, _ = _create_synthetic_sequences(n_samples=10, seq_len=10, seed=99)

        forecaster = LSTMForecaster(epochs=2, batch_size=16,
                                     early_stopping_patience=1)
        forecaster.train(X_train, y_train)
        preds = forecaster.predict(X_test)

        assert preds.shape == (10,)

    def test_predict_before_train_raises(self):
        """Test that predicting before training raises error."""
        from models.lstm_forecaster import LSTMForecaster
        forecaster = LSTMForecaster()
        X_test, _ = _create_synthetic_sequences(n_samples=5, seq_len=10)

        with pytest.raises(RuntimeError, match="not been trained"):
            forecaster.predict(X_test)

    def test_evaluate(self):
        """Test model evaluation returns metrics."""
        from models.lstm_forecaster import LSTMForecaster
        X, y = _create_synthetic_sequences(n_samples=50, seq_len=10)

        forecaster = LSTMForecaster(epochs=2, batch_size=16,
                                     early_stopping_patience=1)
        forecaster.train(X, y)
        metrics = forecaster.evaluate(X[:10], y[:10])

        assert "test_loss" in metrics
        assert "test_mae" in metrics
        assert metrics["test_loss"] >= 0
        assert metrics["test_mae"] >= 0

    def test_training_summary(self):
        """Test training summary generation."""
        from models.lstm_forecaster import LSTMForecaster
        X, y = _create_synthetic_sequences(n_samples=50, seq_len=10)

        forecaster = LSTMForecaster(epochs=2, batch_size=16,
                                     early_stopping_patience=1)
        forecaster.train(X, y)
        summary = forecaster.get_training_summary()

        assert summary["epochs_trained"] > 0
        assert "final_loss" in summary
        assert "best_loss" in summary
