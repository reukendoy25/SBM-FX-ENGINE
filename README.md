# SBM FX Engine

An intelligent FX anomaly detection and rate forecasting system built for the State Bank of Mauritius. It combines unsupervised machine learning (Isolation Forest), deep learning (LSTM networks), and financial NLP (FinBERT) to monitor exchange rate behaviour, flag unusual market events, and generate short-term rate forecasts.

---

## Overview

The engine addresses three core problems in institutional FX risk management:

1. **Anomaly detection** — Identifies abnormal rate movements (flash crashes, liquidity crises, post-event volatility) by learning the normal statistical structure of multi-currency FX data using an Isolation Forest ensemble.

2. **Rate forecasting** — Predicts next-day exchange rates through a hybrid approach: a stacked LSTM captures quantitative temporal patterns from 60-day lookback windows, while a FinBERT sentiment model extracts directional signals from central bank policy communications.

3. **Serving** — Exposes both capabilities via a Flask REST API with an interactive web dashboard for visual exploration. Designed for containerised deployment.

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/sbm-fx-engine.git
cd sbm-fx-engine
pip install -r requirements.txt
```

For the full ML stack (LSTM + FinBERT):

```bash
pip install tensorflow transformers torch
```

### Training

This fetches live FX data from the ECB API, engineers features, trains both models, and saves artifacts:

```bash
python engine/train.py
```

You can also train selectively:

```bash
python engine/train.py --anomaly-only    # Isolation Forest only
python engine/train.py --lstm-only       # LSTM only
python engine/train.py --use-cache       # skip data re-fetch
```

### Running the Server

```bash
python engine/app.py
```

The dashboard opens at [http://localhost:5000](http://localhost:5000). API endpoints are available at the same host.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Flask REST API + UI     │
                    │   localhost:5000            │
                    └──────────┬──────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
 ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
 │ Isolation Forest │ │     LSTM       │  │    FinBERT      │
 │ anomaly detector │ │  forecaster    │  │  sentiment      │
 │ (200 trees)      │ │  (128→64)      │  │  analyzer       │
 └────────┬─────────┘ └───────┬────────┘  └────────┬────────┘
          │                   │                    │
          │                   └──────┬─────────────┘
          │                          │
          │                  ┌────────▼────────┐
          │                  │    Ensemble     │
          │                  │ α=0.7 combiner  │
          │                  └─────────────────┘
          │
 ┌────────▼──────────────────────────────────────────────┐
 │              Feature Engineering Pipeline             │
 │  log returns · rolling vol · RSI · Bollinger · spreads│
 └───────────────────────┬───────────────────────────────┘
                         │
 ┌───────────────────────▼───────────────────────────────┐
 │                  Data Aggregation                     │
 │        ECB API  ·  Yahoo Finance  ·  Curated Texts    │
 └───────────────────────────────────────────────────────┘
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive dashboard |
| `GET` | `/health` | Model status and service health |
| `POST` | `/anomaly` | Detect anomalies in FX data |
| `POST` | `/predict` | Forecast rates (LSTM + sentiment) |
| `POST` | `/sentiment` | Analyse text sentiment via FinBERT |
| `POST` | `/retrain` | Trigger model retraining |

### Example: Sentiment Analysis

```bash
curl -X POST http://localhost:5000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The Federal Reserve raised rates by 75 basis points."}'
```

```json
{
  "status": "success",
  "data": {
    "sentiment": -0.73,
    "label": "negative",
    "confidence": 0.91
  }
}
```

---

## Project Structure

```
sbm-fx-engine/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── engine/
│   ├── app.py                  # Flask API + dashboard serving
│   ├── config.py               # All hyperparameters and paths
│   ├── train.py                # End-to-end training pipeline
│   ├── gunicorn.conf.py        # Production WSGI config
│   ├── templates/
│   │   └── index.html          # Dashboard UI
│   ├── data/
│   │   ├── ecb_fetcher.py      # ECB REST API client
│   │   ├── yfinance_fetcher.py # Macro indicator fetcher
│   │   ├── text_scraper.py     # Curated central bank corpus
│   │   └── preprocessing.py    # Feature engineering
│   ├── models/
│   │   ├── anomaly_detector.py # Isolation Forest wrapper
│   │   ├── lstm_forecaster.py  # Stacked LSTM (Keras)
│   │   ├── finbert_sentiment.py# FinBERT inference
│   │   └── ensemble.py         # Forecast combiner
│   ├── visualization/
│   │   └── anomaly_plots.py    # Matplotlib chart generators
│   └── tests/
│       ├── test_anomaly.py
│       ├── test_api.py
│       └── test_forecaster.py
├── artifacts/                   # Serialised models (generated)
├── data_cache/                  # Cached datasets (generated)
└── plots/                       # Saved charts (generated)
```

---

## Technical Details

### Anomaly Detection

The Isolation Forest is trained on engineered features derived from multi-currency FX rates:

- Log returns and rolling volatility (5, 10, 21, 63-day windows)
- RSI momentum (14-day)
- Bollinger Band width
- Inter-currency spread features

The model uses 200 estimators with a 1% contamination prior. Detected anomalies align with known macro events: the Brexit referendum (Jun 2016), COVID market crash (Mar 2020), Russia-Ukraine invasion (Feb 2022), and others.

### Rate Forecasting

The LSTM takes 60-day input sequences through two recurrent layers (128 and 64 units) with 20% dropout, followed by a dense prediction head. Training uses early stopping (patience=10) and learning rate scheduling.

FinBERT ([ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)) provides sentiment scores from central bank communications. The ensemble combines both signals with a weighted average (default 70% LSTM, 30% sentiment).

### Data Sources

| Source | What | Key |
|--------|------|-----|
| [ECB Data Portal](https://data.ecb.europa.eu/) | Daily FX rates (8 currency pairs vs EUR) | None required |
| Yahoo Finance | Treasury yield, VIX, S&P 500, Gold, Oil | None required |
| Curated corpus | 26 FOMC/ECB communications (2014–2024) | Embedded in source |

### Currency Pairs

USD, GBP, JPY, CHF, AUD, ZAR, INR, MUR — all quoted against EUR.

---

## Testing

```bash
pytest engine/tests/ -v
```

---

## Deployment

The project includes Docker support for containerised deployment:

```bash
docker build -t sbm-fx-engine .
docker run -p 5000:5000 sbm-fx-engine
```

Or with docker-compose:

```bash
docker-compose up
```

In production, the app runs behind Gunicorn with auto-scaled workers and extended timeouts for model inference.

---

## References

- Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008. Isolation forest. *ICDM*.
- Araci, D., 2019. FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.
- Hochreiter, S. and Schmidhuber, J., 1997. Long short-term memory. *Neural Computation*.

---

## License

MIT
