# -*- mode: python ; coding: utf-8 -*-
"""
chess_ai.spec: Universal Multi-Platform PyInstaller Spec File.
Packages Chess AI into standalone binaries for macOS (.app & .dmg), Windows (.exe), and Linux.
"""

import sys
import os
from pathlib import Path

block_cipher = None
PROJECT_ROOT = Path(SPECPATH).parent

datas = [
    (str(PROJECT_ROOT / "bin"), "bin"),
    (str(PROJECT_ROOT / "data"), "data"),
    (str(PROJECT_ROOT / "extension"), "extension"),
    (str(PROJECT_ROOT / "config"), "config"),
]

hidden_imports = [
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "chess",
    "chess.engine",
    "chess.polyglot",
    "chess.svg",
    "edge_tts",
    "loguru",
    "scipy",
    "scipy.interpolate",
    "scipy.stats",
    "numpy",
    "cv2",
    "mss",
    "websockets",
    "asyncio",
    "wave",
    "struct",
    "core.engine_manager",
    "core.hybrid_brain",
    "core.maia_engine",
    "core.opening_book",
    "core.state_machine",
    "core.concept_coach",
    "vision.calibrator",
    "vision.board_detector",
    "audio.voice_assistant",
    "audio.stealth_audio",
    "screen.screen_board_detector",
    "screen.screen_capture",
    "screen.screen_move_detector",
    "online.dom_bridge_server",
    "online.dom_parsers",
    "online.input_router",
    "automation.wind_mouse",
    "automation.mouse_controller",
    "automation.human_behavior",
    "automation.telemetry_validator",
    "ui.ar_drawer",
    "ui.overlay_window",
    "ui.master_dashboard",
    "ui.mini_pill_hud",
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "IPython", "notebook", "matplotlib.tests"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="ChessAI",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="ChessAI",
    )
    app = BUNDLE(
        coll,
        name="ChessAI.app",
        icon=None,
        bundle_identifier="com.chessai.multimodal",
        info_plist={
            "NSHighResolutionCapable": "True",
            "NSCameraUsageDescription": "Chess AI uses the camera to detect physical chessboard moves.",
            "NSMicrophoneUsageDescription": "Chess AI uses audio output for voice assistant features.",
            "CFBundleShortVersionString": "2.0.0",
            "CFBundleVersion": "2.0.0",
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="ChessAI",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
