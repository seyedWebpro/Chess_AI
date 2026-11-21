"""
mini_pill_hud.py: Floating Dynamic Island & Mini-Pill HUD (Phase 12).
A sleek, ultra-compact glassmorphic floating pill widget that unobtrusively displays
win probability, best move, and live engine status in the corner of the screen,
expanding into an interactive tactical card on hover.
"""

from typing import Optional, Dict, Any, Tuple
import math
from PyQt6.QtWidgets import (
    QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout,
    QFrame, QPushButton, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QPoint, QRect, QPropertyAnimation, QEasingCurve, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QPainterPath, QMouseEvent, QEnterEvent
)
from loguru import logger


class MiniPillHUD(QWidget):
    """
    Floating glassmorphism Dynamic Island widget.
    Sits persistently on top of all windows (Chess.com, Lichess, Desktop).
    """

    COLLAPSED_WIDTH = 260
    COLLAPSED_HEIGHT = 44
    EXPANDED_WIDTH = 340
    EXPANDED_HEIGHT = 190

    mode_changed_signal = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        x: int = 40,
        y: int = 40,
        headless: bool = False,
    ):
        super().__init__(parent)
        self.headless = headless
        self.is_expanded = False
        self._drag_pos: Optional[QPoint] = None
        self.click_through = False

        # Live State Data
        self.best_move_san: str = "e4"
        self.best_move_uci: str = "e2e4"
        self.eval_str: str = "+0.3"
        self.win_chance: float = 52.0
        self.is_mate: bool = False
        self.concept_title: str = "تسلط بر مرکز"
        self.concept_explanation: str = "پیشروی پیاده جهت تسلط بر خانه‌های مرکزی و باز کردن قطر فیل."
        self.stealth_mode_name: str = "EARCON"

        # Window Flags: Frameless, Always-On-Top, Transparent Background
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SubWindow
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.setGeometry(x, y, self.COLLAPSED_WIDTH, self.COLLAPSED_HEIGHT)
        self._init_ui()

    def _init_ui(self):
        """Initializes internal widget layouts and labels."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 6, 10, 6)
        self.main_layout.setSpacing(6)

        # Header Pill Row (Always Visible)
        self.header_row = QHBoxLayout()
        self.header_row.setContentsMargins(0, 0, 0, 0)
        self.header_row.setSpacing(8)

        # Win Chance Badge
        self.win_badge = QLabel("52%")
        self.win_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.win_badge.setFixedSize(46, 26)
        self.win_badge.setStyleSheet(
            "background-color: #10b98122; color: #10b981; font-weight: bold; "
            "font-size: 11px; border-radius: 13px; border: 1px solid #10b98155;"
        )
        self.header_row.addWidget(self.win_badge)

        # Move Label
        self.move_label = QLabel("♟ e4")
        self.move_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.move_label.setStyleSheet("color: #ffffff; letter-spacing: 0.5px;")
        self.header_row.addWidget(self.move_label)

        # Eval Label
        self.eval_label = QLabel("+0.3")
        self.eval_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.eval_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.eval_label.setStyleSheet("color: #38bdf8;")
        self.header_row.addWidget(self.eval_label)

        # Status Indicator Dot
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #22c55e; font-size: 11px;")
        self.header_row.addWidget(self.status_dot)

        self.main_layout.addLayout(self.header_row)

        # Extended Details Container (Shown when Expanded)
        self.details_container = QWidget()
        self.details_layout = QVBoxLayout(self.details_container)
        self.details_layout.setContentsMargins(4, 4, 4, 4)
        self.details_layout.setSpacing(6)

        # Concept Title
        self.concept_title_label = QLabel("🎯 تسلط بر مرکز")
        self.concept_title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.concept_title_label.setStyleSheet("color: #facc15;")
        self.details_layout.addWidget(self.concept_title_label)

        # Concept Explanation
        self.concept_desc_label = QLabel(self.concept_explanation)
        self.concept_desc_label.setFont(QFont("Segoe UI", 9))
        self.concept_desc_label.setStyleSheet("color: #cbd5e1; line-height: 1.3;")
        self.concept_desc_label.setWordWrap(True)
        self.details_layout.addWidget(self.concept_desc_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.mode_btn = QPushButton("🔊 Audio: Earcon")
        self.mode_btn.setFixedHeight(24)
        self.mode_btn.setStyleSheet(
            "background: #1e293b; color: #94a3b8; font-size: 10px; "
            "border-radius: 6px; border: 1px solid #334155; padding: 0 8px;"
        )
        self.mode_btn.clicked.connect(self._toggle_audio_mode)
        btn_row.addWidget(self.mode_btn)

        self.click_thru_btn = QPushButton("🖱 Click-Thru: OFF")
        self.click_thru_btn.setFixedHeight(24)
        self.click_thru_btn.setStyleSheet(
            "background: #1e293b; color: #94a3b8; font-size: 10px; "
            "border-radius: 6px; border: 1px solid #334155; padding: 0 8px;"
        )
        self.click_thru_btn.clicked.connect(self._toggle_click_through)
        btn_row.addWidget(self.click_thru_btn)

        self.details_layout.addLayout(btn_row)
        self.main_layout.addWidget(self.details_container)

        self.details_container.setVisible(False)

    def update_state(
        self,
        best_move_san: str,
        best_move_uci: str,
        eval_str: str,
        win_chance: float,
        concept_title: str = "",
        concept_explanation: str = "",
        is_mate: bool = False,
    ):
        """Updates HUD state in real-time."""
        self.best_move_san = best_move_san
        self.best_move_uci = best_move_uci
        self.eval_str = eval_str
        self.win_chance = max(0.0, min(100.0, float(win_chance)))
        self.is_mate = is_mate
        if concept_title:
            self.concept_title = concept_title
        if concept_explanation:
            self.concept_explanation = concept_explanation

        # Format move with chess piece glyph
        glyph = self._get_piece_glyph(best_move_san)
        self.move_label.setText(f"{glyph} {best_move_san}")
        self.eval_label.setText(eval_str)
        self.win_badge.setText(f"{int(round(self.win_chance))}%")

        # Colorize win badge according to advantage
        if self.win_chance >= 60.0:
            color = "#10b981"
            bg = "#10b98122"
        elif self.win_chance <= 40.0:
            color = "#ef4444"
            bg = "#ef444422"
        else:
            color = "#f59e0b"
            bg = "#f59e0b22"

        self.win_badge.setStyleSheet(
            f"background-color: {bg}; color: {color}; font-weight: bold; "
            f"font-size: 11px; border-radius: 13px; border: 1px solid {color}55;"
        )

        if self.concept_title:
            self.concept_title_label.setText(f"🎯 {self.concept_title}")
        if self.concept_explanation:
            self.concept_desc_label.setText(self.concept_explanation)

        self.update()

    def _get_piece_glyph(self, san: str) -> str:
        if san.startswith("N"): return "♘"
        if san.startswith("B"): return "♗"
        if san.startswith("R"): return "♖"
        if san.startswith("Q"): return "♕"
        if san.startswith("K"): return "♔"
        if "O-O" in san: return "🏰"
        return "♟"

    def expand_hud(self):
        """Expands HUD into detailed tactical view."""
        if not self.is_expanded:
            self.is_expanded = True
            self.details_container.setVisible(True)
            self.resize(self.EXPANDED_WIDTH, self.EXPANDED_HEIGHT)
            self.update()

    def collapse_hud(self):
        """Collapses HUD back into mini pill."""
        if self.is_expanded:
            self.is_expanded = False
            self.details_container.setVisible(False)
            self.adjustSize()
            self.resize(self.COLLAPSED_WIDTH, self.COLLAPSED_HEIGHT)
            self.update()

    def enterEvent(self, event: QEnterEvent):
        """Expand on mouse enter."""
        if not self.click_through:
            self.expand_hud()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Collapse on mouse leave."""
        if not self.click_through:
            self.collapse_hud()
        super().leaveEvent(event)

    def _toggle_audio_mode(self):
        modes = ["EARCON", "WHISPER", "VOICE", "MUTE"]
        curr_idx = modes.index(self.stealth_mode_name) if self.stealth_mode_name in modes else 0
        next_mode = modes[(curr_idx + 1) % len(modes)]
        self.stealth_mode_name = next_mode
        self.mode_btn.setText(f"🔊 Audio: {next_mode}")
        self.mode_changed_signal.emit(next_mode)

    def _toggle_click_through(self):
        self.click_through = not self.click_through
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.click_through)
        status_txt = "ON" if self.click_through else "OFF"
        self.click_thru_btn.setText(f"🖱 Click-Thru: {status_txt}")
        if self.click_through:
            self.collapse_hud()

    # Dragging Window by Mouse Click & Move
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and not self.click_through:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos and not self.click_through:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None

    def paintEvent(self, event):
        """Draws the sleek glassmorphic pill background with subtle glow."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        radius = 18 if self.is_expanded else 21

        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, rect.width() - 1, rect.height() - 1, radius, radius)

        # Semi-transparent dark frosted glass gradient
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0.0, QColor(13, 17, 23, 235))
        gradient.setColorAt(1.0, QColor(22, 27, 34, 245))

        painter.fillPath(path, gradient)

        # Cyber border with subtle cyan/glow outline
        border_pen = QPen(QColor(56, 189, 248, 90 if self.is_expanded else 45), 1.2)
        painter.strokePath(path, border_pen)
