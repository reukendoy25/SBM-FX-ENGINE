"""
SBM FX Engine — Flask REST API
================================
RESTful API exposing the Isolation Forest anomaly detector and
LSTM rate forecaster.

FinBERT is available as a standalone /sentiment endpoint for
ad-hoc text analysis but is NOT part of the /predict pipeline.

Endpoints:
  GET  /health    → Health check and model status
  POST /anomaly   → FX anomaly detection
  POST /predict   → Rate forecasting (LSTM)
  POST /sentiment → FinBERT sentiment analysis (standalone)
  POST /retrain   → Trigger model retraining
"""

import logging
import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import config
from models.anomaly_detector import AnomalyDetector
from models.lstm_forecaster import LSTMForecaster
from models.ensemble import EnsembleForecaster

# ============================================================
# App Initialization
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sbm_fx_engine")

app = Flask(__name__, template_folder="templates")
CORS(app)

# ============================================================
# Model Loading
# ============================================================

# Global model references
_anomaly_model = None
_lstm_model = None
_ensemble_model = None
_sentiment_model = None  # lazy-loaded only when /sentiment is called

_models_loaded = {
    "anomaly_detector": False,
    "lstm_forecaster": False,
    "ensemble": False,
    # finbert_sentiment is intentionally omitted — loaded on-demand in /sentiment
}


def _load_models():
    """Load serialized models on startup."""
    global _anomaly_model, _lstm_model, _ensemble_model

    # Anomaly Detector
    try:
        if os.path.exists(config.ANOMALY_MODEL_PATH):
            _anomaly_model = AnomalyDetector.load()
            _models_loaded["anomaly_detector"] = True
            logger.info("✓ Anomaly detector loaded.")
        else:
            logger.warning("Anomaly model not found — train first.")
    except Exception as e:
        logger.error(f"Failed to load anomaly model: {e}")

    # LSTM Forecaster
    try:
        if os.path.exists(config.LSTM_MODEL_PATH):
            _lstm_model = LSTMForecaster.load()
            _models_loaded["lstm_forecaster"] = True
            logger.info("✓ LSTM forecaster loaded.")
        else:
            logger.warning("LSTM model not found — train first.")
    except Exception as e:
        logger.error(f"Failed to load LSTM model: {e}")

    # Ensemble (LSTM-only wrapper)
    try:
        _ensemble_model = EnsembleForecaster()
        _models_loaded["ensemble"] = True
        logger.info("✓ Ensemble forecaster ready (LSTM-only).")
    except Exception as e:
        logger.error(f"Failed to init ensemble: {e}")

    # NOTE: FinBERT is NOT loaded here — it is lazy-loaded on /sentiment calls.


# Load on import
_load_models()


# ============================================================
# Helper Functions
# ============================================================

def _error_response(message: str, status_code: int = 400) -> tuple:
    """Create standardized error response."""
    return jsonify({
        "status": "error",
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }), status_code


def _success_response(data: dict, status_code: int = 200) -> tuple:
    """Create standardized success response."""
    return jsonify({
        "status": "success",
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }), status_code


# ============================================================
# API Endpoints
# ============================================================

@app.route("/", methods=["GET"])
def dashboard():
    """Serve the interactive dashboard UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint — returns model status."""
    return _success_response({
        "service": "SBM FX Engine",
        "version": "1.0.0",
        "models": _models_loaded,
        "all_models_loaded": all(_models_loaded.values()),
    })


@app.route("/anomaly", methods=["POST"])
def detect_anomalies():
    """
    Detect anomalies in FX transaction data.

    Expected JSON payload:
    {
        "data": {
            "USD": [1.12, 1.13, ...],
            "GBP": [0.86, 0.87, ...],
            ...
        },
        "dates": ["2024-01-01", "2024-01-02", ...]  // optional
    }
    """
    if not _models_loaded["anomaly_detector"]:
        return _error_response(
            "Anomaly model not loaded. Run training first.", 503
        )

    try:
        payload = request.get_json()
        if not payload or "data" not in payload:
            return _error_response("Missing 'data' field in request.")

        # Parse input data
        fx_data = pd.DataFrame(payload["data"])
        if "dates" in payload:
            fx_data.index = pd.to_datetime(payload["dates"])

        # Build features
        from data.preprocessing import AnomalyFeaturePipeline
        pipeline = AnomalyFeaturePipeline()
        features = pipeline.build_features(fx_data)

        if features.empty:
            return _error_response(
                "Insufficient data for feature engineering. "
                "Need at least 63 data points."
            )

        # Predict
        results = _anomaly_model.predict(features)

        # Format response
        anomalies = results[results["anomaly_label"] == -1]
        response = {
            "total_samples": len(results),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": round(len(anomalies) / len(results) * 100, 2),
            "anomaly_dates": [
                str(d.date()) if hasattr(d, "date") else str(d)
                for d in anomalies.index
            ],
            "scores": {
                str(d.date()) if hasattr(d, "date") else str(d): {
                    "label": int(row["anomaly_label"]),
                    "score": round(float(row["anomaly_score_normalized"]), 4),
                }
                for d, row in results.iterrows()
            },
        }

        return _success_response(response)

    except Exception as e:
        logger.error(f"Anomaly detection error: {traceback.format_exc()}")
        return _error_response(f"Processing error: {str(e)}", 500)


@app.route("/predict", methods=["POST"])
def predict_rate():
    """
    Forecast FX rate using the LSTM model.

    Expected JSON payload:
    {
        "sequences": [[...], ...],  // LSTM input sequences (60-day window)
        "current_rate": 1.12        // Optional: current rate for direction
    }
    """
    if not _models_loaded["lstm_forecaster"]:
        return _error_response(
            "LSTM model not loaded. Run training first.", 503
        )

    try:
        payload = request.get_json()
        if not payload:
            return _error_response("Empty request body.")

        if "sequences" not in payload:
            return _error_response("Missing 'sequences' field.")

        sequences = np.array(payload["sequences"])
        if sequences.ndim == 2:
            sequences = sequences.reshape(1, *sequences.shape)
        lstm_pred = float(_lstm_model.predict(sequences)[0])

        current_rate = payload.get("current_rate")
        result = _ensemble_model.predict_single(
            lstm_pred=lstm_pred,
            current_rate=current_rate,
        )

        return _success_response(result)

    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return _error_response(f"Prediction error: {str(e)}", 500)


@app.route("/sentiment", methods=["POST"])
def analyze_sentiment():
    """
    Standalone FinBERT sentiment analysis for financial text.
    This endpoint is independent of the forecasting pipeline.

    Expected JSON payload:
    {
        "text": "The Federal Reserve raised rates..."
        // OR
        "texts": ["text1", "text2", ...]
    }
    """
    # Lazy-load FinBERT only when this endpoint is actually called
    global _sentiment_model
    if _sentiment_model is None:
        try:
            from models.finbert_sentiment import FinBERTSentimentAnalyzer
            _sentiment_model = FinBERTSentimentAnalyzer()
            logger.info("FinBERT lazy-loaded on /sentiment request.")
        except Exception as e:
            return _error_response(f"FinBERT unavailable: {str(e)}", 503)

    try:
        payload = request.get_json()
        if not payload:
            return _error_response("Empty request body.")

        if "text" in payload:
            result = _sentiment_model.analyze(payload["text"])
            return _success_response(result)
        elif "texts" in payload:
            results = _sentiment_model.analyze_batch(payload["texts"])
            return _success_response({"results": results})
        else:
            return _error_response("Missing 'text' or 'texts' field.")

    except Exception as e:
        logger.error(f"Sentiment error: {traceback.format_exc()}")
        return _error_response(f"Sentiment analysis error: {str(e)}", 500)


@app.route("/retrain", methods=["POST"])
def retrain_models():
    """
    Trigger model retraining.

    Expected JSON payload:
    {
        "models": ["anomaly", "lstm", "all"]  // which models to retrain
    }
    """
    try:
        payload = request.get_json() or {}
        models_to_train = payload.get("models", ["all"])

        # Import training logic
        from train import run_training

        results = run_training(
            train_anomaly="anomaly" in models_to_train or "all" in models_to_train,
            train_lstm="lstm" in models_to_train or "all" in models_to_train,
        )

        # Reload models
        _load_models()

        return _success_response({
            "message": "Retraining complete.",
            "results": results,
        })

    except Exception as e:
        logger.error(f"Retraining error: {traceback.format_exc()}")
        return _error_response(f"Retraining error: {str(e)}", 500)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG,
    )
