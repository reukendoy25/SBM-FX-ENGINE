"""
Autoencoder Anomaly Detector
==============================
Uses a shallow Keras autoencoder to detect FX anomalies via
reconstruction error. The model is trained exclusively on normal
data; unseen anomalies produce higher reconstruction errors.

Architecture:
  Input → Dense(32, relu) → Dense(16, relu) → Dense(32, relu) → Output
  Loss: Mean Squared Error

Threshold: set at the (1 - contamination) percentile of
training reconstruction errors.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector for FX market data.

    Anomaly scores:
      - Prediction: -1 = anomaly, 1 = normal
      - anomaly_score: raw reconstruction MSE
      - anomaly_score_normalized: 0 (normal) to 1 (anomalous)
    """

    def __init__(self, **kwargs):
        params = {**config.AUTOENCODER_PARAMS, **kwargs}
        self.params = params
        self.model = None
        self.scaler = None
        self.threshold = None
        self.is_fitted = False
        self.feature_names = None
        self._input_dim = None
        logger.info(f"AutoencoderDetector initialized with params: {params}")

    def _build_model(self, input_dim: int):
        """Build the autoencoder architecture."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "TensorFlow is required for AutoencoderDetector. "
                "Install with: pip install tensorflow"
            )

        enc_dims = self.params.get("encoding_dims", [32, 16])
        lr = self.params.get("learning_rate", 1e-3)

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        # Encoder
        for dim in enc_dims:
            x = keras.layers.Dense(dim, activation="relu")(x)

        # Decoder (mirror)
        for dim in reversed(enc_dims[:-1]):
            x = keras.layers.Dense(dim, activation="relu")(x)

        outputs = keras.layers.Dense(input_dim, activation="linear")(x)

        model = keras.Model(inputs, outputs, name="fx_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )
        return model

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE between input and reconstruction."""
        reconstructed = self.model.predict(X, verbose=0)
        errors = np.mean((X - reconstructed) ** 2, axis=1)
        return errors

    def train(self, features: pd.DataFrame) -> "AutoencoderDetector":
        """
        Train the autoencoder on the feature matrix.

        Args:
            features: DataFrame of engineered features

        Returns:
            self (for chaining)
        """
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"Training Autoencoder on {features.shape[0]} samples "
            f"with {features.shape[1]} features..."
        )

        self.feature_names = list(features.columns)
        self._input_dim = features.shape[1]

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values).astype("float32")

        # Build and train model
        self.model = self._build_model(self._input_dim)

        epochs = self.params.get("epochs", 50)
        batch_size = self.params.get("batch_size", 32)
        val_split = self.params.get("validation_split", 0.1)

        try:
            import tensorflow as tf
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
            self.model.fit(
                X, X,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=[early_stop],
                verbose=0,
            )
        except Exception as e:
            logger.warning(f"Early stopping not used due to: {e}. Fitting without.")
            self.model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)

        # Set threshold from training error distribution
        train_errors = self._reconstruction_errors(X)
        contamination = config.AUTOENCODER_PARAMS.get("contamination", 0.01)
        self.threshold = float(np.percentile(train_errors, (1 - contamination) * 100))

        self.is_fitted = True

        n_anomalies = (train_errors > self.threshold).sum()
        anomaly_pct = n_anomalies / len(train_errors) * 100
        logger.info(
            f"Autoencoder training complete. Threshold={self.threshold:.6f}. "
            f"Flagged {n_anomalies} anomalies ({anomaly_pct:.2f}%) in training data."
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomaly labels and scores.

        Args:
            features: DataFrame of engineered features

        Returns:
            DataFrame with anomaly_label, anomaly_score, anomaly_score_normalized
        """
        if not self.is_fitted:
            raise RuntimeError(
                "AutoencoderDetector has not been trained. Call train() first."
            )

        X = self.scaler.transform(features.values).astype("float32")
        errors = self._reconstruction_errors(X)

        labels = np.where(errors > self.threshold, -1, 1)

        # Normalize reconstruction error to [0, 1]
        err_min = errors.min()
        err_max = errors.max()
        err_range = err_max - err_min if err_max != err_min else 1.0
        normalized = (errors - err_min) / err_range

        results = pd.DataFrame({
            "anomaly_label": labels,
            "anomaly_score": errors,       # higher = more anomalous
            "anomaly_score_normalized": normalized,
        }, index=features.index)

        n_anomalies = (labels == -1).sum()
        logger.info(
            f"Autoencoder prediction: {n_anomalies} anomalies detected "
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
        """Save model weights and scaler."""
        base_path = path or config.AUTOENCODER_MODEL_PATH
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an untrained Autoencoder model.")

        # Save Keras model separately
        self.model.save(base_path)

        meta = {
            "scaler": self.scaler,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "input_dim": self._input_dim,
            "params": self.params,
            "model_path": base_path,
        }
        joblib.dump(meta, config.AUTOENCODER_SCALER_PATH)
        logger.info(f"AutoencoderDetector saved — weights: {base_path}, meta: {config.AUTOENCODER_SCALER_PATH}")

    @classmethod
    def load(cls, model_path: str = None, meta_path: str = None) -> "AutoencoderDetector":
        """Load a trained model from disk."""
        import tensorflow as tf

        model_path = model_path or config.AUTOENCODER_MODEL_PATH
        meta_path = meta_path or config.AUTOENCODER_SCALER_PATH

        for p in (model_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")

        meta = joblib.load(meta_path)
        detector = cls.__new__(cls)
        detector.params = meta["params"]
        detector.model = tf.keras.models.load_model(model_path)
        detector.scaler = meta["scaler"]
        detector.threshold = meta["threshold"]
        detector.feature_names = meta["feature_names"]
        detector._input_dim = meta["input_dim"]
        detector.is_fitted = True

        logger.info(f"AutoencoderDetector loaded from {model_path}")
        return detector
