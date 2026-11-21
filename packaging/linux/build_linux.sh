#!/usr/bin/env bash
# build_linux.sh: Builds standalone Linux distribution binary and AppImage tarball

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"

echo "🐧 Building Chess AI for Linux..."
cd "${PROJECT_ROOT}"

pyinstaller --clean --noconfirm packaging/chess_ai.spec

if [ -f "${DIST_DIR}/ChessAI" ]; then
    echo "📦 Creating Linux tarball package..."
    cd "${DIST_DIR}"
    tar -czvf "ChessAI-Linux-x86_64.tar.gz" ChessAI/
    echo "✅ Linux package successfully generated at: ${DIST_DIR}/ChessAI-Linux-x86_64.tar.gz"
fi
