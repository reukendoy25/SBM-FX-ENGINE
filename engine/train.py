"""
SBM FX Engine — End-to-End Training Script
============================================
Orchestrates the complete model training pipeline:
1. Fetch FX data from ECB API / yfinance
2. Engineer features
3. Train Isolation Forest anomaly detector
4. Train LSTM rate forecaster
5. Run FinBERT sentiment analysis on central bank texts
6. Serialize all model artifacts

Usage:
    python train.py                    # Train all models
    python train.py --anomaly-only     # Train anomaly detector only
    python train.py --lstm-only        # Train LSTM only
    python train.py --use-cache        # Use cached data (skip fetching)
"""

import argparse
import logging
import os
import sys
import time
import json
import numpy as np
import pandas as pd

import config
from data.ecb_fetcher import ECBFetcher
from data.yfinance_fetcher import YFinanceFetcher
from data.text_scraper import TextScraper
from data.preprocessing import AnomalyFeaturePipeline, LSTMFeaturePipeline
from models.anomaly_detector import IsolationForestDetector, AnomalyDetector, EnsembleAnomalyDetector
from models.ocsvm_detector import OCSVMDetector
from models.autoencoder_detector import AutoencoderDetector
from models.anomaly_evaluator import AnomalyEvaluator
from models.lstm_forecaster import LSTMForecaster
from models.ensemble import EnsembleForecaster
from visualization.anomaly_plots import (
    plot_fx_with_anomalies,
    plot_anomaly_score_distribution,
    plot_correlation_heatmap,
    plot_forecast_results,
    plot_sentiment_timeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

PLOTS_DIR = os.path.join(config.BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def fetch_data(use_cache: bool = False) -> dict:
    """
    Fetch all required data (FX rates, macro indicators, central bank texts).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA AGGREGATION")
    logger.info("=" * 60)

    data = {}

    # --- ECB FX Data ---
    ecb_cache = os.path.join(config.DATA_DIR, "ecb_fx_data.csv")
    if use_cache and os.path.exists(ecb_cache):
        logger.info("Loading ECB FX data from cache...")
        data["fx"] = pd.read_csv(ecb_cache, index_col=0, parse_dates=True)
    else:
        try:
            logger.info("Fetching FX data from ECB API...")
            ecb = ECBFetcher()
            data["fx"] = ecb.fetch_multiple()
            ecb.save_to_cache(data["fx"])
        except Exception as e:
            logger.warning(f"ECB API failed: {e}. Trying yfinance fallback...")
            yf_fetcher = YFinanceFetcher()
            data["fx"] = yf_fetcher.fetch_fx_pairs()
            yf_fetcher.save_to_cache(data["fx"], "ecb_fx_data.csv")

    logger.info(f"FX data shape: {data['fx'].shape}")

    # --- Macro Indicators ---
    macro_cache = os.path.join(config.DATA_DIR, "macro_indicators.csv")
    if use_cache and os.path.exists(macro_cache):
        logger.info("Loading macro indicators from cache...")
        data["macro"] = pd.read_csv(macro_cache, index_col=0, parse_dates=True)
    else:
        try:
            logger.info("Fetching macro indicators from yfinance...")
            yf_fetcher = YFinanceFetcher()
            data["macro"] = yf_fetcher.fetch_macro_indicators()
            yf_fetcher.save_to_cache(data["macro"], "macro_indicators.csv")
        except Exception as e:
            logger.warning(f"Failed to fetch macro indicators: {e}")
            data["macro"] = pd.DataFrame()

    if not data["macro"].empty:
        logger.info(f"Macro data shape: {data['macro'].shape}")

    # --- Central Bank Texts ---
    logger.info("Loading central bank communications...")
    scraper = TextScraper()
    data["texts"] = scraper.get_all()
    scraper.save_to_cache()
    logger.info(f"Loaded {len(data['texts'])} text entries.")

    return data


def _build_detectors(detector_mode: str):
    """
    Build the requested detector(s) based on the CLI --detector flag.

    Returns a dict of {name: detector_instance} for training and comparison.
    The first entry is treated as the primary detector to serialize as the
    active artifact.

    Supported modes:
        if           — Isolation Forest only (default)
        ocsvm        — One-Class SVM only
        ae           — Autoencoder only
        if+ocsvm     — Ensemble of IF + OCSVM
        if+ae        — Ensemble of IF + Autoencoder
        ocsvm+ae     — Ensemble of OCSVM + Autoencoder
        all          — Trains IF, OCSVM, AE individually AND as a full ensemble
    """
    mode = detector_mode.lower().strip()

    if mode == "if":
        det = IsolationForestDetector()
        return {det.name: det}

    if mode == "ocsvm":
        det = OCSVMDetector()
        return {det.name: det}

    if mode == "ae":
        det = AutoencoderDetector()
        return {det.name: det}

    if mode == "if+ocsvm":
        det = EnsembleAnomalyDetector(
            [IsolationForestDetector(), OCSVMDetector()]
        )
        return {det.name: det}

    if mode == "if+ae":
        det = EnsembleAnomalyDetector(
            [IsolationForestDetector(), AutoencoderDetector()]
        )
        return {det.name: det}

    if mode == "ocsvm+ae":
        det = EnsembleAnomalyDetector(
            [OCSVMDetector(), AutoencoderDetector()]
        )
        return {det.name: det}

    if mode == "all":
        return {
            "IsolationForest": IsolationForestDetector(),
            "OneClassSVM": OCSVMDetector(),
            "Autoencoder": AutoencoderDetector(),
            "Ensemble(IF+OCSVM+AE)": EnsembleAnomalyDetector(
                [IsolationForestDetector(), OCSVMDetector(), AutoencoderDetector()]
            ),
        }

    raise ValueError(
        f"Unknown detector mode '{detector_mode}'. "
        "Choose from: if, ocsvm, ae, if+ocsvm, if+ae, ocsvm+ae, all"
    )


def train_anomaly_detector(fx_data: pd.DataFrame,
                           detector_mode: str = "if") -> dict:
    """
    Train the requested anomaly detector(s) and compare via evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info(f"PHASE 2: ANOMALY DETECTION TRAINING  [mode={detector_mode}]")
    logger.info("=" * 60)

    # Build features
    pipeline = AnomalyFeaturePipeline()
    features = pipeline.build_features(fx_data)

    if features.empty:
        logger.error("Feature matrix is empty — insufficient data.")
        return {"status": "failed", "reason": "empty_features"}

    # Build and train detectors
    detectors = _build_detectors(detector_mode)
    all_results = {}

    for name, detector in detectors.items():
        logger.info(f"Training [{name}]...")
        try:
            detector.train(features)
            results = detector.predict(features)
            all_results[name] = results
        except Exception as e:
            logger.error(f"[{name}] training failed: {e}")

    # ── Evaluation & Comparison ───────────────────────────────────────────────
    if all_results:
        contamination = config.IFOREST_PARAMS.get("contamination", 0.01)
        comparison = AnomalyEvaluator.compare(all_results, contamination)

    # ── Primary detector = first one (or IF if 'all' mode) ───────────────────
    primary_name = list(detectors.keys())[0]
    primary_detector = detectors[primary_name]
    primary_results = all_results.get(primary_name)

    if primary_results is None:
        return {"status": "failed", "reason": "primary_detector_failed"}

    anomalies = primary_results[primary_results["anomaly_label"] == -1]

    # Show top anomaly dates from primary detector
    top_anomalies = primary_detector.get_anomaly_dates(features, fx_data)
    if not top_anomalies.empty:
        logger.info(f"\nTop 10 anomalous dates [{primary_name}]:")
        for date, row in top_anomalies.head(10).iterrows():
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            logger.info(
                f"  {date_str} | Score: {row['anomaly_score_normalized']:.4f}"
            )

    # Feature importance (IF only)
    if hasattr(primary_detector, "get_feature_importance"):
        importance = primary_detector.get_feature_importance(features, n_top=10)
        logger.info(f"\nTop 10 feature importances:\n{importance}")

    # Generate plots (using primary detector)
    for currency in fx_data.columns[:3]:
        try:
            plot_fx_with_anomalies(
                fx_data, primary_results, currency,
                save_path=os.path.join(PLOTS_DIR, f"anomaly_{currency}.png"),
            )
        except Exception as e:
            logger.warning(f"Failed to plot {currency}: {e}")

    try:
        plot_anomaly_score_distribution(
            primary_results,
            save_path=os.path.join(PLOTS_DIR, "anomaly_distribution.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot distribution: {e}")

    try:
        plot_correlation_heatmap(
            fx_data,
            save_path=os.path.join(PLOTS_DIR, "correlation_heatmap.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot heatmap: {e}")

    # Serialize primary detector using its own path
    try:
        primary_detector.serialize()
        logger.info(f"Primary detector [{primary_name}] serialized.")
    except Exception as e:
        logger.warning(f"Serialization of [{primary_name}] failed: {e}")

    return {
        "status": "success",
        "detector_mode": detector_mode,
        "primary_detector": primary_name,
        "total_samples": len(features),
        "anomalies_detected": int(len(anomalies)),
        "anomaly_rate_pct": round(len(anomalies) / len(features) * 100, 2),
    }


def train_lstm_forecaster(fx_data: pd.DataFrame,
                           macro_data: pd.DataFrame = None,
                           target_currency: str = "USD") -> dict:
    """
    Train the LSTM rate forecaster.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: LSTM FORECASTER TRAINING")
    logger.info("=" * 60)

    # Build features and sequences
    pipeline = LSTMFeaturePipeline()
    features = pipeline.build_features(fx_data, macro_data, target_currency)

    if len(features) < config.LSTM_PARAMS["sequence_length"] + 10:
        logger.error("Insufficient data for LSTM training.")
        return {"status": "failed", "reason": "insufficient_data"}

    X, y, dates = pipeline.create_sequences(features, target_currency)
    split = pipeline.split_train_test(X, y, dates)

    logger.info(f"Train set: {split['X_train'].shape}")
    logger.info(f"Test set: {split['X_test'].shape}")

    # Train model
    forecaster = LSTMForecaster()
    history = forecaster.train(
        split["X_train"], split["y_train"],
        split["X_test"], split["y_test"],
    )

    # Evaluate
    metrics = forecaster.evaluate(split["X_test"], split["y_test"])

    # Generate predictions for visualization
    test_preds = forecaster.predict(split["X_test"])
    target_idx = list(features.columns).index(target_currency)

    # Inverse transform
    actual = pipeline.inverse_transform_predictions(
        split["y_test"], features.shape[1], target_idx
    )
    predicted = pipeline.inverse_transform_predictions(
        test_preds, features.shape[1], target_idx
    )

    # Plot
    try:
        plot_forecast_results(
            actual, predicted,
            dates=split["dates_test"],
            currency=target_currency,
            save_path=os.path.join(PLOTS_DIR, f"forecast_{target_currency}.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot forecast: {e}")

    # Serialize
    forecaster.serialize()

    # Save scaler
    import joblib
    joblib.dump(pipeline.scaler, config.LSTM_SCALER_PATH)
    logger.info(f"LSTM model saved to {config.LSTM_MODEL_PATH}")
    logger.info(f"Scaler saved to {config.LSTM_SCALER_PATH}")

    # Save training summary
    summary = forecaster.get_training_summary()
    summary.update(metrics)

    return {
        "status": "success",
        "target_currency": target_currency,
        "model_path": config.LSTM_MODEL_PATH,
        **summary,
    }


def run_sentiment_analysis(texts_df: pd.DataFrame) -> dict:
    """
    Run FinBERT sentiment analysis on central bank texts.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3b: FINBERT SENTIMENT ANALYSIS")
    logger.info("=" * 60)

    analyzer = FinBERTSentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(texts_df)

    logger.info(f"\nSentiment Analysis Results:")
    logger.info(f"  Total texts analyzed: {len(sentiment_df)}")
    logger.info(f"  Positive: {(sentiment_df['label'] == 'positive').sum()}")
    logger.info(f"  Negative: {(sentiment_df['label'] == 'negative').sum()}")
    logger.info(f"  Neutral: {(sentiment_df['label'] == 'neutral').sum()}")
    logger.info(f"  Mean sentiment: {sentiment_df['sentiment'].mean():.4f}")

    # Plot
    try:
        plot_sentiment_timeline(
            sentiment_df,
            save_path=os.path.join(PLOTS_DIR, "sentiment_timeline.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot sentiment: {e}")

    # Save results
    sentiment_path = os.path.join(config.DATA_DIR, "sentiment_results.csv")
    sentiment_df.to_csv(sentiment_path)

    return {
        "status": "success",
        "total_analyzed": len(sentiment_df),
        "mean_sentiment": round(float(sentiment_df["sentiment"].mean()), 4),
        "results_path": sentiment_path,
    }


def run_training(train_anomaly: bool = True,
                train_lstm: bool = True,
                use_cache: bool = False,
                detector_mode: str = "if") -> dict:
    """
    Run the full training pipeline.
    """
    start_time = time.time()
    results = {}

    # Fetch data
    data = fetch_data(use_cache=use_cache)

    # Train anomaly detector
    if train_anomaly:
        results["anomaly"] = train_anomaly_detector(data["fx"], detector_mode=detector_mode)

    # Train LSTM
    if train_lstm:
        results["lstm"] = train_lstm_forecaster(
            data["fx"], data["macro"], target_currency="USD"
        )

    # Run sentiment analysis as standalone (not part of forecasting)
    # Note: FinBERT is no longer wired into the ensemble predictor.
    # It is available via the /sentiment API endpoint only.
    # To run a standalone sentiment pass, call run_sentiment_analysis() directly.

    # Save ensemble config
    ensemble = EnsembleForecaster()
    ensemble.save_config()

    elapsed = time.time() - start_time
    results["total_time_seconds"] = round(elapsed, 2)

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE in {elapsed:.1f}s")
    logger.info("=" * 60)

    # Save training results
    results_path = os.path.join(config.ARTIFACTS_DIR, "training_results.json")
    with open(results_path, "w") as f:
        # Convert non-serializable types
        json.dump(
            {k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v
             for k, v in results.items()},
            f, indent=2, default=str,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBM FX Engine Training")
    parser.add_argument("--anomaly-only", action="store_true",
                        help="Train anomaly detector only")
    parser.add_argument("--lstm-only", action="store_true",
                        help="Train LSTM forecaster only")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached data instead of fetching")
    parser.add_argument(
        "--detector", default="if",
        choices=["if", "ocsvm", "ae", "if+ocsvm", "if+ae", "ocsvm+ae", "all"],
        help=(
            "Anomaly detector to use: "
            "if=IsolationForest (default), "
            "ocsvm=OneClassSVM, ae=Autoencoder, "
            "if+ocsvm/if+ae/ocsvm+ae=ensembles, all=train+compare all"
        ),
    )
    args = parser.parse_args()

    train_anomaly = not args.lstm_only
    train_lstm = not args.anomaly_only

    results = run_training(
        train_anomaly=train_anomaly,
        train_lstm=train_lstm,
        use_cache=args.use_cache,
        detector_mode=args.detector,
    )

    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
