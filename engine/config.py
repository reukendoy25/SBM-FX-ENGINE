"""
SBM FX Engine - Centralized Configuration
==========================================
All constants, hyperparameters, and configuration values for the engine.
"""

import os
from datetime import datetime

# ============================================================
# Data Sources
# ============================================================

# ECB Statistical Data Warehouse REST API
ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"
ECB_API_FORMAT = "csvdata"

# Currency pairs to track (vs EUR)
CURRENCY_PAIRS = ["USD", "GBP", "JPY", "CHF", "AUD", "ZAR", "INR", "MUR"]

# Historical data range
DATA_START_DATE = "2014-01-01"
DATA_END_DATE = datetime.now().strftime("%Y-%m-%d")

# yfinance supplementary tickers
YFINANCE_TICKERS = {
    "treasury_10y": "^TNX",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "gold": "GC=F",
    "oil": "CL=F",
}

# FX pairs for yfinance fallback (vs USD)
YFINANCE_FX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDZAR": "USDZAR=X",
    "USDINR": "USDINR=X",
}

# ============================================================
# Anomaly Detection - Isolation Forest
# ============================================================

IFOREST_PARAMS = {
    "n_estimators": 200,
    "contamination": 0.01,
    "max_samples": "auto",
    "random_state": 42,
    "n_jobs": -1,
}

# Feature engineering windows
ROLLING_WINDOWS = [5, 10, 21, 63]  # 1wk, 2wk, 1mo, 3mo
VOLATILITY_WINDOW = 21
MOMENTUM_WINDOW = 14

# ============================================================
# Anomaly Detection - One-Class SVM
# ============================================================

OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": 0.01,       # upper bound on training anomaly fraction (≈ contamination)
    "gamma": "scale", # RBF bandwidth: 1 / (n_features * X.var())
}

# ============================================================
# Anomaly Detection - Autoencoder
# ============================================================

AUTOENCODER_PARAMS = {
    "encoding_dims": [32, 16],   # encoder layer sizes (decoder mirrors)
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.1,
    "contamination": 0.01,       # threshold percentile = (1 - contamination) * 100
}

# ============================================================
# LSTM Forecaster
# ============================================================

LSTM_PARAMS = {
    "sequence_length": 60,         # lookback window (trading days)
    "lstm_units_1": 128,           # first LSTM layer units
    "lstm_units_2": 64,            # second LSTM layer units
    "dense_units": 32,             # dense layer units
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.1,
    "early_stopping_patience": 10,
}

# Train/Test split ratio
TRAIN_TEST_SPLIT = 0.8

# ============================================================
# FinBERT Sentiment Analysis
# ============================================================

FINBERT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 512

# Sentiment score mapping
SENTIMENT_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}

# ============================================================
# Ensemble Forecaster
# ============================================================

ENSEMBLE_ALPHA = 0.7  # Weight for LSTM (1 - alpha for sentiment)

# ============================================================
# Flask API
# ============================================================

API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# ============================================================
# Paths
# ============================================================

# engine/ is one level below project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data_cache")

# Ensure directories exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model artifact filenames
ANOMALY_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl")
OCSVM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ocsvm_detector.pkl")
AUTOENCODER_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "autoencoder.keras")
AUTOENCODER_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "autoencoder_meta.pkl")
LSTM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lstm_forecaster.keras")
LSTM_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "lstm_scaler.pkl")
ENSEMBLE_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "ensemble_config.json")
