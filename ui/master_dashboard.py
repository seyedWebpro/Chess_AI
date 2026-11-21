"""
master_dashboard.py: Master Unified Control Dashboard & GUI (Phase 10).
Integrates all 10 phases of the Chess AI system: Webcam MVP, Screen CV, DOM Bridge,
Transparent Overlay, Hybrid Brain (Maia + Polyglot), WindMouse Anti-Cheat Bot, and Persian Voice.
"""

from typing import Optional, Dict, Any, List
import sys
import chess
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QProgressBar,
    QTextEdit, QFrame, QSplitter
)
from PyQt6.QtGui import QFont, QColor, QPalette
from loguru import logger

from core.hybrid_brain import HybridBrain
from core.maia_engine import Persona
from automation.mouse_controller import HumanMouseController, MockMouseController
from automation.telemetry_validator import TelemetryValidator


class MasterDashboardWindow(QMainWindow):
    """
    Unified PyQt6 Master Control Dashboard for the Multi-Modal Chess AI System.
    """

    move_executed_signal = pyqtSignal(dict)

    def __init__(
        self,
        hybrid_brain: Optional[HybridBrain] = None,
        mouse_controller: Optional[HumanMouseController] = None,
        headless: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("Chess AI - Master Control Dashboard (Phases 1-10)")
        self.resize(1100, 750)
        self.headless = headless

        self.brain = hybrid_brain or HybridBrain(persona=Persona.MAIA_1500, use_opening_book=True)
        self.mouse = mouse_controller or HumanMouseController(backend=MockMouseController(fast_mode=True), fast_mode=True)
        self.board = chess.Board()

        self.last_analysis: Dict[str, Any] = {}
        self.last_telemetry: Dict[str, Any] = {}

        self._setup_dark_theme()
        self._init_ui()

    def _setup_dark_theme(self) -> None:
        """Applies a professional dark cyber aesthetic."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(15, 17, 23))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(240, 243, 246))
        palette.setColor(QPalette.ColorRole.Base, QColor(24, 27, 36))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(32, 36, 48))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(240, 243, 246))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(240, 243, 246))
        palette.setColor(QPalette.ColorRole.Text, QColor(240, 243, 246))
        palette.setColor(QPalette.ColorRole.Button, QColor(35, 40, 55))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(240, 243, 246))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(56, 189, 248))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(15, 17, 23))
        self.setPalette(palette)

    def _init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left Column: Controls & Configuration
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setStyleSheet("background-color: #141721; border-radius: 12px; border: 1px solid #282f42;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        # Title
        title_label = QLabel("♟️ CHESS AI MASTER CONTROL")
        title_label.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #38bdf8; letter-spacing: 0.5px;")
        left_layout.addWidget(title_label)

        # Section 1: Brain & Persona
        sec1_title = QLabel("🧠 DUAL-BRAIN & PERSONA (PHASE 10)")
        sec1_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        sec1_title.setStyleSheet("color: #94a3b8;")
        left_layout.addWidget(sec1_title)

        self.persona_combo = QComboBox()
        self.persona_combo.setStyleSheet("background: #1e2333; color: white; padding: 6px; border-radius: 6px; border: 1px solid #384259;")
        for p in Persona:
            self.persona_combo.addItem(p.value, p)
        self.persona_combo.setCurrentIndex(2) # Maia 1500
        self.persona_combo.currentIndexChanged.connect(self._on_persona_changed)
        left_layout.addWidget(self.persona_combo)

        self.book_checkbox = QCheckBox("Enable Polyglot Opening Book (0ms Theory)")
        self.book_checkbox.setChecked(True)
        self.book_checkbox.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        self.book_checkbox.toggled.connect(self._on_book_toggled)
        left_layout.addWidget(self.book_checkbox)

        # Section 2: Automation & Anti-Cheat
        sec2_title = QLabel("🖱️ AUTOMATION & ANTI-CHEAT (PHASE 9)")
        sec2_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        sec2_title.setStyleSheet("color: #94a3b8;")
        left_layout.addWidget(sec2_title)

        self.autobot_checkbox = QCheckBox("Autonomous Bot Mode (WindMouse Physics)")
        self.autobot_checkbox.setChecked(False)
        self.autobot_checkbox.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        left_layout.addWidget(self.autobot_checkbox)

        self.drag_checkbox = QCheckBox("Prefer Drag & Drop (instead of Click-Click)")
        self.drag_checkbox.setChecked(False)
        self.drag_checkbox.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        left_layout.addWidget(self.drag_checkbox)

        # Section 3: Audio & Persian Voice Assistant
        sec3_title = QLabel("🔊 PERSIAN VOICE ASSISTANT (PHASE 5)")
        sec3_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        sec3_title.setStyleSheet("color: #94a3b8;")
        left_layout.addWidget(sec3_title)

        self.voice_checkbox = QCheckBox("Persian Neural TTS Voice (Farid / Dilara)")
        self.voice_checkbox.setChecked(True)
        self.voice_checkbox.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        left_layout.addWidget(self.voice_checkbox)

        # Trigger Move Button
        self.btn_trigger_move = QPushButton("⚡ Evaluate Position & Play Move")
        self.btn_trigger_move.setStyleSheet("background: #0284c7; color: white; font-weight: bold; padding: 10px; border-radius: 8px; font-size: 13px;")
        self.btn_trigger_move.clicked.connect(self.trigger_ai_move)
        left_layout.addWidget(self.btn_trigger_move)

        left_layout.addStretch()

        # System Status Indicators
        status_box = QFrame()
        status_box.setStyleSheet("background: #0d1017; border-radius: 8px; padding: 8px; border: 1px solid #1f2636;")
        status_layout = QVBoxLayout(status_box)
        self.lbl_perception = QLabel("Perception: Screen CV (2290 FPS) & DOM (Ready)")
        self.lbl_perception.setStyleSheet("color: #34d399; font-size: 11px;")
        self.lbl_overlay = QLabel("Overlay: PyQt6 Transparent Canvas (Click-Through: 270 FPS)")
        self.lbl_overlay.setStyleSheet("color: #f472b6; font-size: 11px;")
        status_layout.addWidget(self.lbl_perception)
        status_layout.addWidget(self.lbl_overlay)
        left_layout.addWidget(status_box)

        main_layout.addWidget(left_panel, stretch=4)

        # Right Column: Live Analysis, Eval Bar & Telemetry Audit
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setStyleSheet("background-color: #141721; border-radius: 12px; border: 1px solid #282f42;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)

        # Top Live Telemetry Badge
        self.lbl_eval_score = QLabel("+0.00")
        self.lbl_eval_score.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        self.lbl_eval_score.setStyleSheet("color: #22c55e;")
        right_layout.addWidget(self.lbl_eval_score)

        self.lbl_best_move = QLabel("Best Move: Ready")
        self.lbl_best_move.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        self.lbl_best_move.setStyleSheet("color: #f8fafc;")
        right_layout.addWidget(self.lbl_best_move)

        # Win Chance Progress Bar
        self.win_bar = QProgressBar()
        self.win_bar.setRange(0, 100)
        self.win_bar.setValue(50)
        self.win_bar.setTextVisible(True)
        self.win_bar.setFormat("Win Chance: %p%")
        self.win_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 6px;
                text-align: center;
                color: white;
                background-color: #1e293b;
                height: 20px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #22c55e;
                border-radius: 5px;
            }
        """)
        right_layout.addWidget(self.win_bar)

        # Live Event / Persian Voice Log
        log_title = QLabel("📋 LIVE EVENT & TELEMETRY STREAM")
        log_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        log_title.setStyleSheet("color: #94a3b8;")
        right_layout.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #090d16; color: #cbd5e1; font-family: monospace; font-size: 11px; border-radius: 6px; border: 1px solid #1e293b;")
        right_layout.addWidget(self.log_text)

        main_layout.addWidget(right_panel, stretch=6)

    def _on_persona_changed(self, index: int) -> None:
        persona = self.persona_combo.currentData()
        if persona:
            self.brain.set_persona(persona)
            self._log(f"Switched persona to: {persona.value}")

    def _on_book_toggled(self, checked: bool) -> None:
        self.brain.set_opening_book_enabled(checked)
        self._log(f"Polyglot opening book: {'ENABLED' if checked else 'DISABLED'}")

    def _log(self, text: str) -> None:
        self.log_text.append(f"> {text}")

    def set_position(self, fen: str) -> None:
        """Sets board position FEN."""
        self.board.set_fen(fen)
        self._log(f"Position set: {fen[:35]}...")

    def trigger_ai_move(self) -> Dict[str, Any]:
        """Calculates and executes AI move according to active Hybrid Brain."""
        decision = self.brain.decide_move(self.board, time_limit=0.10)
        self.last_analysis = decision

        move = decision.get("move")
        if move:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            uci = decision.get("uci", f"{from_sq}{to_sq}")

            eval_str = decision.get("eval_str", "+0.00")
            win_pct = decision.get("win_chance_pct", 50.0)

            self.lbl_eval_score.setText(eval_str)
            self.lbl_best_move.setText(f"Move: {uci} ({from_sq} &rarr; {to_sq}) &bull; Source: {decision.get('source')}")
            self.win_bar.setValue(int(win_pct))

            self._log(f"AI Decision: {uci} | Source: {decision.get('source')} | Deliberation: {decision.get('deliberation_time_sec'):.2f}s")

            # Execute via WindMouse if autobot enabled
            if self.autobot_checkbox.isChecked():
                board_bbox = (100, 100, 800, 800)
                is_drag = self.drag_checkbox.isChecked()
                mouse_report = self.mouse.play_move(
                    from_square=from_sq,
                    to_square=to_sq,
                    board_bbox=board_bbox,
                    is_drag=is_drag,
                )
                self.last_telemetry = mouse_report
                self._log(f"WindMouse Bot Executed: {uci} ({mouse_report['mode']}) in {mouse_report['total_trajectory_points']} pts")

            self.board.push(move)
            self.move_executed_signal.emit(decision)

        return decision

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Returns programmatic state of dashboard for test assertions."""
        return {
            "fen": self.board.fen(),
            "persona": self.persona_combo.currentData().name,
            "opening_book_enabled": self.book_checkbox.isChecked(),
            "autobot_enabled": self.autobot_checkbox.isChecked(),
            "last_analysis": self.last_analysis,
            "last_telemetry": self.last_telemetry,
        }

    def closeEvent(self, event) -> None:
        if self.brain:
            self.brain.close()
        event.accept()
