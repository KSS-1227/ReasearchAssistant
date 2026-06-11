#!/bin/bash
# -*- coding: utf-8 -*-
# Entrypoint - starts FastAPI backend then Streamlit frontend

echo "======================================"
echo "Starting Research Assistant..."
echo "======================================"

# Load .env if present
if [ -f /app/.env ]; then
    set -a
    source /app/.env
    set +a
fi

# Graceful shutdown handler
cleanup() {
    echo "Shutting down services..."
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGTERM SIGINT

# Start FastAPI backend (1 worker - safe for containers)
echo "Starting FastAPI on port 8000..."
python -m uvicorn fastapi_app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &
FASTAPI_PID=$!

# Wait for FastAPI to be ready (up to 60s)
echo "Waiting for FastAPI to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "FastAPI ready (${i}s)"
        break
    fi
    if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "FastAPI process died. Exiting."
        exit 1
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

# Start Streamlit frontend (replace shell with Streamlit)
echo "Starting Streamlit on port 8501..."
exec streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=info
