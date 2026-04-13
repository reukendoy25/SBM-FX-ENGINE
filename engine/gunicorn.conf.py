"""
Gunicorn Configuration
======================
Production WSGI server configuration for the Flask API.
"""

import multiprocessing

# Server socket
bind = "0.0.0.0:5000"

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)
worker_class = "sync"
worker_connections = 1000

# Timeouts (extended for model inference)
timeout = 120
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "sbm-fx-engine"

# Preload app to share model memory across workers
preload_app = True
