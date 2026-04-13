"""
Ensemble Forecaster
===================
Wraps the LSTM forecaster to produce a final rate prediction.

FinBERT sentiment was previously combined here via a weighted average
but was removed — the 26-text corpus was too small to produce a
statistically validated signal worth including in live predictions.

The /sentiment API endpoint still exists for standalone text analysis.
"""

import logging
import json
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Thin wrapper around the LSTM forecaster that produces a final
    rate prediction with directional metadata.

    Kept as a class (rather than removing entirely) for:
      - API compatibility with /predict endpoint
      - Future extensibility if a validated signal is added
    """

    def __init__(self):
        logger.info("EnsembleForecaster initialized (LSTM-only mode).")

    def predict_single(self, lstm_pred: float,
                       current_rate: float = None) -> dict:
        """
        Produce a single rate prediction from the LSTM output.

        Args:
            lstm_pred:    LSTM predicted rate
            current_rate: Current market rate (optional)

        Returns:
            Dict with prediction and directional metadata.
        """
        result = {
            "prediction": float(lstm_pred),
            "lstm_prediction": float(lstm_pred),
            "model": "LSTM",
        }

        if current_rate is not None:
            change = lstm_pred - current_rate
            change_pct = (change / current_rate) * 100
            result.update({
                "current_rate": float(current_rate),
                "predicted_change": float(change),
                "predicted_change_pct": float(round(change_pct, 4)),
                "direction": "up" if change > 0 else "down" if change < 0 else "flat",
            })

        return result

    def combine(self, lstm_predictions: np.ndarray,
                current_rates: np.ndarray = None) -> dict:
        """
        Produce batch predictions from an array of LSTM outputs.

        Args:
            lstm_predictions: Array of LSTM predicted rates
            current_rates:    Optional array of current market rates

        Returns:
            Dict with predictions and directional metadata.
        """
        preds = np.asarray(lstm_predictions, dtype=float)

        if current_rates is not None:
            curr = np.asarray(current_rates, dtype=float)[-len(preds):]
            direction = np.sign(preds - curr)
            confidence = np.abs(preds - curr) / (curr + 1e-10)
        else:
            direction = np.zeros_like(preds)
            confidence = np.zeros_like(preds)

        result = {
            "predictions": preds.tolist(),
            "direction": direction.tolist(),
            "directional_confidence": confidence.tolist(),
            "model": "LSTM",
            "n_predictions": len(preds),
        }

        logger.info(f"EnsembleForecaster: {len(preds)} predictions generated.")
        return result

    def save_config(self, path: str = None):
        """Save ensemble configuration."""
        path = path or config.ENSEMBLE_CONFIG_PATH
        cfg = {"model": "LSTM", "version": "2.0"}
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Ensemble config saved to {path}")

    @classmethod
    def load_config(cls, path: str = None) -> "EnsembleForecaster":
        """Load ensemble configuration."""
        return cls()
