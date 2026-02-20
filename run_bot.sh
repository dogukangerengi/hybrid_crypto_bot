#!/bin/bash

PROJECT_DIR="$HOME/hybrid_crypto_bot"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
SRC_DIR="$PROJECT_DIR/src"
LOG_DIR="$PROJECT_DIR/logs/system"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/bot_$(date +%Y-%m-%d).log"

echo "========================================" >> "$LOG_FILE"
echo "BOT BAŞLADI: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$SRC_DIR"
"$VENV_PYTHON" main.py \
    --live \
    --top 5 \
    >> "$LOG_FILE" 2>&1

echo "BOT BİTTİ: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
