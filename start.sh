#!/bin/bash

# 0. Download the model before starting anything
echo "📥 Downloading model from Hugging Face Hub..."
uv run python -m scripts.download_model_from_hf

# 1. Start FastAPI in the background
echo "🚀 Starting FastAPI..."
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 2. Wait a few seconds for the API to initialize
echo "⏳ Waiting for API to launch..."
sleep 5

# 3. Start Streamlit in the foreground (Port 7860 for HF Spaces)
echo "🚀 Starting Streamlit..."
uv run streamlit run src/app/main.py --server.port 7860 --server.address 0.0.0.0
