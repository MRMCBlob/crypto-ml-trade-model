#!/bin/bash
set -e

echo "========================================"
echo "Crypto ML Trading System - Starting"
echo "========================================"

# Check if models exist
MODELS_DIR="/app/models/saved"
STORAGE_DIR="/app/storage"

# Step 1: Download data if not present or if forced
if [ ! -f "$STORAGE_DIR/trading.db" ] || [ "$FORCE_DATA_DOWNLOAD" = "true" ]; then
    echo ""
    echo "[1/3] Downloading historical data..."
    python scripts/download_multi_coin_data.py
else
    echo ""
    echo "[1/3] Data already exists, skipping download (set FORCE_DATA_DOWNLOAD=true to re-download)"
fi

# Step 2: Train models if not present or if forced
MODEL_COUNT=$(find "$MODELS_DIR" -name "*.keras" -o -name "*.joblib" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -lt 2 ] || [ "$FORCE_MODEL_TRAINING" = "true" ]; then
    echo ""
    echo "[2/3] Training ML models (this may take a while)..."
    python scripts/train_multi_coin_models.py
else
    echo ""
    echo "[2/3] Models already exist ($MODEL_COUNT files), skipping training (set FORCE_MODEL_TRAINING=true to re-train)"
fi

# Step 3: Start trading
echo ""
echo "[3/3] Starting continuous trading with 12-hour reports..."
echo "========================================"
python scripts/run_multi_coin_trading.py --continuous
