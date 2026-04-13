FROM python:3.9-slim

# Metadata
LABEL maintainer="SBM FX Engine" \
    description="AI-powered FX Anomaly Detection & Rate Forecasting"

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts data_cache plots

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run with Gunicorn from the engine package
CMD ["gunicorn", "--config", "engine/gunicorn.conf.py", "engine.app:app"]
