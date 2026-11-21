#!/usr/bin/env bash
# ==============================================================================
# Helper to verify or configure Stockfish engine
# ==============================================================================
set -e

echo "🔍 Checking for Stockfish chess engine..."

if [ -f "bin/stockfish" ] && [ -x "bin/stockfish" ]; then
    echo "✅ Local Stockfish binary detected at bin/stockfish"
    exit 0
fi

if command -v stockfish >/dev/null 2>&1; then
    SYS_PATH=$(command -v stockfish)
    echo "✅ System Stockfish detected at $SYS_PATH"
    exit 0
fi

echo "⚠️ Stockfish was not found in ./bin/stockfish or system PATH."
echo "Please install it using your platform's package manager:"
echo "  - macOS:   brew install stockfish"
echo "  - Ubuntu:  sudo apt-get install stockfish"
echo "  - Windows: Download from https://stockfishchess.org/download/"
exit 1
