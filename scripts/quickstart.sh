#!/usr/bin/env bash
# ==============================================================================
# Chess AI - One-Click Automated Setup & Diagnostic
# ==============================================================================
set -e

echo "♟️ Initializing Chess AI Environment..."

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "⬇️ Installing Python dependencies..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet

if [ -f "bin/stockfish" ]; then
    chmod +x bin/stockfish
fi

echo "🧪 Running fast sanity check..."
.venv/bin/pytest tests/test_engine.py tests/test_maia_onnx.py -q

echo "✅ Environment ready! Launching Master Graphical Dashboard..."
.venv/bin/python3 main.py --gui
