#!/bin/bash

# 1. Start FastAPI in the background
# We use 'nohup' or just '&' to send it to the background.
# Host 0.0.0.0 is important for internal networking within the container.
# Port 8000 is where Streamlit will look for the API.
echo "🚀 Starting FastAPI..."
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds for the API to initialize
# This prevents Streamlit from showing an error immediately upon load.
echo "⏳ Waiting for API to launch..."
sleep 5

# 3. Start Streamlit in the foreground
# Streamlit must run on port 7860 for Hugging Face Spaces to detect it.
echo "🚀 Starting Streamlit..."
uv run streamlit run src/app/main.py --server.port 7860 --server.address 0.0.0.0
