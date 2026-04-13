"""
Tests — Flask API Endpoints
============================
Integration tests for the Flask REST API.
"""

import pytest
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create a Flask test client."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Health response should have correct structure."""
        response = client.get("/health")
        data = response.get_json()
        assert data["status"] == "success"
        assert "data" in data
        assert data["data"]["service"] == "SBM FX Engine"
        assert "models" in data["data"]
        assert "timestamp" in data

    def test_health_models_status(self, client):
        """Health should report model load status."""
        response = client.get("/health")
        data = response.get_json()
        models = data["data"]["models"]
        assert "anomaly_detector" in models
        assert "lstm_forecaster" in models
        assert "finbert_sentiment" in models
        assert "ensemble" in models


class TestSentimentEndpoint:
    """Tests for the /sentiment endpoint."""

    def test_sentiment_empty_body(self, client):
        """Empty body should return error."""
        response = client.post(
            "/sentiment",
            data=json.dumps({}),
            content_type="application/json",
        )
        data = response.get_json()
        assert response.status_code == 400 or data["status"] == "error"

    def test_sentiment_missing_text(self, client):
        """Missing text field should return error."""
        response = client.post(
            "/sentiment",
            data=json.dumps({"wrong_field": "test"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["status"] == "error"


class TestAnomalyEndpoint:
    """Tests for the /anomaly endpoint."""

    def test_anomaly_missing_data(self, client):
        """Missing data field should return error."""
        response = client.post(
            "/anomaly",
            data=json.dumps({}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["status"] == "error"

    def test_anomaly_insufficient_data(self, client):
        """Insufficient data points should return error or message."""
        response = client.post(
            "/anomaly",
            data=json.dumps({
                "data": {"USD": [1.12, 1.13, 1.14]},
            }),
            content_type="application/json",
        )
        # Should either be an error or a 503 (model not loaded)
        assert response.status_code in [400, 500, 503]


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_empty_body(self, client):
        """Empty body should return error."""
        response = client.post(
            "/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["status"] == "error"

    def test_predict_missing_sequences(self, client):
        """Missing sequences should return error."""
        response = client.post(
            "/predict",
            data=json.dumps({"text": "test"}),
            content_type="application/json",
        )
        data = response.get_json()
        # Either error for missing sequences or 503 for model not loaded
        assert response.status_code in [400, 500, 503]
