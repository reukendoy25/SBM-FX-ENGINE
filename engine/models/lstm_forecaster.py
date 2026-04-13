"""
LSTM Rate Forecaster
====================
Implements a stacked LSTM network for multivariate FX rate forecasting
using TensorFlow/Keras.

Architecture:
  Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
        → Dense(32, ReLU) → Dense(1)
"""

import logging
import os
import numpy as np
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def _build_lstm_model(input_shape: tuple, params: dict):
    """Build the LSTM model architecture using Keras."""
    # Lazy import to reduce startup time
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # First LSTM layer (returns sequences for stacking)
        layers.LSTM(
            units=params["lstm_units_1"],
            return_sequences=True,
            input_shape=input_shape,
        ),
        layers.Dropout(params["dropout_rate"]),

        # Second LSTM layer
        layers.LSTM(
            units=params["lstm_units_2"],
            return_sequences=False,
        ),
        layers.Dropout(params["dropout_rate"]),

        # Dense layers
        layers.Dense(params["dense_units"], activation="relu"),
        layers.Dense(1),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    return model


class LSTMForecaster:
    """
    Stacked LSTM network for FX rate prediction.

    Uses a 2-layer LSTM architecture with dropout regularization
    to capture long-term temporal dependencies in financial time series.
    """

    def __init__(self, **kwargs):
        """
        Initialize the forecaster.

        Default params from config can be overridden via kwargs.
        """
        self.params = {**config.LSTM_PARAMS, **kwargs}
        self.model = None
        self.history = None
        self.is_fitted = False
        logger.info(f"LSTMForecaster initialized with params: {self.params}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the LSTM model.

        Args:
            X_train: Training sequences, shape (samples, seq_len, features)
            y_train: Training targets, shape (samples,)
            X_val: Optional validation sequences
            y_val: Optional validation targets

        Returns:
            Training history dict with loss/metric curves.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info(
            f"Training LSTM: {X_train.shape[0]} samples, "
            f"sequence_length={input_shape[0]}, "
            f"n_features={input_shape[1]}"
        )

        # Build model
        self.model = _build_lstm_model(input_shape, self.params)
        self.model.summary(print_fn=logger.info)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.params["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # Validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_data=validation_data,
            validation_split=(
                self.params["validation_split"]
                if validation_data is None else 0.0
            ),
            callbacks=callbacks,
            verbose=1,
        )

        self.is_fitted = True
        final_loss = self.history.history["loss"][-1]
        logger.info(f"Training complete. Final loss: {final_loss:.6f}")

        return self.history.history

    def predict(self, X_input: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.

        Args:
            X_input: Input sequences, shape (samples, seq_len, features)

        Returns:
            Predictions array, shape (samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        predictions = self.model.predict(X_input, verbose=0)
        return predictions.flatten()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data.

        Returns:
            Dict with loss and MAE metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained.")

        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {"test_loss": loss, "test_mae": mae}
        logger.info(f"Test metrics: {metrics}")
        return metrics

    def serialize(self, model_path: str = None, scaler_path: str = None):
        """
        Save the trained model and associated artifacts.

        Args:
            model_path: Path for the Keras model
            scaler_path: Path for the scaler (saved separately)
        """
        model_path = model_path or config.LSTM_MODEL_PATH
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an untrained model.")

        self.model.save(model_path)
        logger.info(f"LSTM model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str = None) -> "LSTMForecaster":
        """Load a trained model from disk."""
        import tensorflow as tf

        model_path = model_path or config.LSTM_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        forecaster = cls.__new__(cls)
        forecaster.params = config.LSTM_PARAMS.copy()
        forecaster.model = tf.keras.models.load_model(model_path)
        forecaster.history = None
        forecaster.is_fitted = True

        logger.info(f"LSTM model loaded from {model_path}")
        return forecaster

    def get_training_summary(self) -> dict:
        """Return a summary of the training process."""
        if self.history is None:
            return {"status": "not_trained"}

        history = self.history.history
        return {
            "epochs_trained": len(history["loss"]),
            "final_loss": history["loss"][-1],
            "best_loss": min(history["loss"]),
            "final_val_loss": (
                history.get("val_loss", [None])[-1]
            ),
            "best_val_loss": (
                min(history["val_loss"]) if "val_loss" in history else None
            ),
        }
