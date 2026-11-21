"""
build_dist.py: Universal Multi-Platform Build & Distribution Pipeline.
Compiles Chess AI into standalone distributions:
- macOS: .app bundle & drag-and-drop .dmg installer
- Windows: standalone .exe & Inno Setup configuration
- Linux: standalone executable and tarball package
"""

import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SPEC_FILE = PROJECT_ROOT / "packaging" / "chess_ai.spec"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def check_pyinstaller_installed():
    try:
        import PyInstaller
        logger.info(f"PyInstaller version {PyInstaller.__version__} detected.")
        return True
    except ImportError:
        logger.error("PyInstaller is not installed in the current environment.")
        logger.info("Install with: pip install pyinstaller")
        return False


def build_current_platform(create_dmg: bool = True):
    """Executes PyInstaller build for the current host OS."""
    if not check_pyinstaller_installed():
        sys.exit(1)

    logger.info(f"Starting compilation using spec: {SPEC_FILE}...")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(SPEC_FILE),
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"PyInstaller build failed with exit code: {result.returncode}")
        return False

    logger.success("PyInstaller compilation succeeded!")

    # macOS DMG Packaging
    if sys.platform == "darwin" and create_dmg:
        app_path = DIST_DIR / "ChessAI.app"
        if app_path.exists():
            logger.info("Generating macOS DMG drag-and-drop installer...")
            dmg_script = PROJECT_ROOT / "packaging" / "macos" / "create_dmg.sh"
            subprocess.run(["bash", str(dmg_script)], cwd=str(PROJECT_ROOT))
        else:
            logger.warning(f"Could not find {app_path} to build DMG.")

    # Linux Tarball Packaging
    elif sys.platform.startswith("linux"):
        linux_script = PROJECT_ROOT / "packaging" / "linux" / "build_linux.sh"
        subprocess.run(["bash", str(linux_script)], cwd=str(PROJECT_ROOT))

    logger.success(f"Build artifacts available in: {DIST_DIR}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Chess AI Cross-Platform Distribution Builder")
    parser.add_argument("--no-dmg", action="store_true", help="Skip DMG creation on macOS")
    args = parser.parse_args()

    success = build_current_platform(create_dmg=not args.no_dmg)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
