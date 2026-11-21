#!/usr/bin/env bash
# create_dmg.sh: Creates a macOS drag-and-drop .dmg installer using native hdiutil.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
APP_PATH="${DIST_DIR}/ChessAI.app"
DMG_PATH="${DIST_DIR}/ChessAI-macOS-Installer.dmg"
VOLUME_NAME="Chess AI 2.0"
DMG_TEMP_DIR="${DIST_DIR}/dmg_temp"

if [ ! -d "${APP_PATH}" ]; then
    echo "❌ Error: ${APP_PATH} does not exist. Run PyInstaller first."
    exit 1
fi

echo "📦 Packaging ${APP_PATH} into ${DMG_PATH}..."

# Clean old dmg and temp directory
rm -f "${DMG_PATH}"
rm -rf "${DMG_TEMP_DIR}"
mkdir -p "${DMG_TEMP_DIR}"

# Copy .app and create Applications symlink
cp -R "${APP_PATH}" "${DMG_TEMP_DIR}/"
ln -s /Applications "${DMG_TEMP_DIR}/Applications"

# Create disk image
hdiutil create -volname "${VOLUME_NAME}" -srcfolder "${DMG_TEMP_DIR}" -ov -format UDZO "${DMG_PATH}"

# Cleanup temp directory
rm -rf "${DMG_TEMP_DIR}"

echo "✅ DMG Installer successfully generated at: ${DMG_PATH}"
