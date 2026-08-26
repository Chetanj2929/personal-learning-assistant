#!/bin/bash
set -e

# Start FastAPI backend in background on port 8000 (internal only)
uvicorn main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Give uvicorn a moment to start before Streamlit tries to connect
sleep 3

# Start Streamlit on port 7860 (the port HF Spaces exposes publicly)
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false

# If Streamlit exits, kill uvicorn too
kill 
