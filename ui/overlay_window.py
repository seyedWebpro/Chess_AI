"""
overlay_window.py: Transparent & Click-Through Screen Overlay using PyQt6.
Renders AR neon move arrows, evaluation bars, and glassmorphism HUD directly over
online chessboards (Chess.com, Lichess) without intercepting mouse clicks.
"""

from typing import Optional, Tuple, Dict, Any, List
import math
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPolygonF,
    QFont,
    QLinearGradient,
    QPainterPath,
    QPixmap,
)
from loguru import logger


FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]


class TransparentOverlayWindow(QWidget):
    """
    Frameless, transparent, click-through overlay window.
    Sits persistently above the desktop and draws visual recommendations directly
    over the chessboard.
    """

    def __init__(
        self,
        board_bbox: Optional[Tuple[int, int, int, int]] = None,
        white_bottom: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.board_bbox = board_bbox  # (x, y, width, height)
        self.white_bottom = white_bottom

        # Overlay State
        self.best_move: Optional[str] = None          # e.g. "e2e4"
        self.best_move_san: Optional[str] = None      # e.g. "e4" or "Qxf7#"
        self.from_square: Optional[str] = None        # "e2"
        self.to_square: Optional[str] = None          # "e4"
        self.eval_score: Optional[float] = 0.0        # in pawns e.g. +1.5
        self.is_mate: bool = False
        self.mate_in: Optional[int] = None
        self.win_chance: float = 50.0
        self.threat_squares: List[str] = []

        # Visibility Toggles
        self.show_arrows: bool = True
        self.show_eval_bar: bool = True
        self.show_hud: bool = True
        self.show_threats: bool = True
        self.click_through: bool = True

        self._init_window_flags()

    def _init_window_flags(self):
        """Initializes frameless, translucent, stays-on-top, and click-through window properties."""
        # 1. Window Flags: Frameless, Always on Top, Tool window (no dock icon stealing)
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)

        # 2. Translucent Background (100% transparent canvas)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 3. Click-Through: Mouse events pass directly to windows underneath
        self.set_click_through(self.click_through)

        # 4. Show without stealing focus from active browser window
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def set_click_through(self, enabled: bool):
        """Enables or disables click-through behavior."""
        self.click_through = enabled
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        logger.info(f"Overlay click-through set to: {enabled}")

    def set_board_geometry(self, x: int, y: int, width: int, height: int, white_bottom: bool = True):
        """Sets the target chessboard location on the screen."""
        self.board_bbox = (int(x), int(y), int(width), int(height))
        self.white_bottom = white_bottom
        self.update()
        logger.info(f"Overlay board geometry updated: bbox={self.board_bbox}, white_bottom={white_bottom}")

    def update_analysis(self, analysis: Dict[str, Any]):
        """Updates overlay visual state from engine analysis dictionary."""
        self.best_move = analysis.get("best_move")
        self.best_move_san = analysis.get("best_move_san")
        self.from_square = analysis.get("from_square")
        self.to_square = analysis.get("to_square")
        self.eval_score = analysis.get("eval_score", 0.0)
        self.is_mate = analysis.get("is_mate", False)
        self.mate_in = analysis.get("mate_in")
        self.win_chance = analysis.get("win_chance_percent", 50.0)

        threats = analysis.get("threat_squares", [])
        self.threat_squares = list(threats) if threats else []

        self.update()

    def clear_overlay(self):
        """Clears all recommendations and reset overlay to blank state."""
        self.best_move = None
        self.best_move_san = None
        self.from_square = None
        self.to_square = None
        self.eval_score = 0.0
        self.is_mate = False
        self.mate_in = None
        self.threat_squares.clear()
        self.update()

    def square_to_center_point(self, sq_name: str) -> Optional[QPointF]:
        """Calculates screen pixel center coordinates for a square e.g. 'e4'."""
        if not self.board_bbox:
            return None

        sq = sq_name.lower().strip()
        if len(sq) != 2 or sq[0] not in FILES or sq[1] not in RANKS:
            return None

        bx, by, bw, bh = self.board_bbox
        sq_size = bw / 8.0

        col = FILES.index(sq[0]) if self.white_bottom else (7 - FILES.index(sq[0]))
        rank_idx = int(sq[1]) - 1
        row = (7 - rank_idx) if self.white_bottom else rank_idx

        cx = bx + col * sq_size + sq_size / 2.0
        cy = by + row * sq_size + sq_size / 2.0

        return QPointF(cx, cy)

    def square_to_rect(self, sq_name: str) -> Optional[QRectF]:
        """Returns the bounding QRectF for a square on screen."""
        if not self.board_bbox:
            return None

        sq = sq_name.lower().strip()
        if len(sq) != 2 or sq[0] not in FILES or sq[1] not in RANKS:
            return None

        bx, by, bw, bh = self.board_bbox
        sq_size = bw / 8.0

        col = FILES.index(sq[0]) if self.white_bottom else (7 - FILES.index(sq[0]))
        rank_idx = int(sq[1]) - 1
        row = (7 - rank_idx) if self.white_bottom else rank_idx

        x = bx + col * sq_size
        y = by + row * sq_size

        return QRectF(x, y, sq_size, sq_size)

    def paintEvent(self, event):
        """Custom QPainter rendering loop."""
        if not self.board_bbox:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Threat / Check squares
        if self.show_threats and self.threat_squares:
            for sq in self.threat_squares:
                rect = self.square_to_rect(sq)
                if rect:
                    self._draw_glow_square(painter, rect, QColor(239, 68, 68, 70), QColor(239, 68, 68, 220))

        # 2. From and To square highlights
        if self.from_square:
            rect_from = self.square_to_rect(self.from_square)
            if rect_from:
                self._draw_glow_square(painter, rect_from, QColor(6, 182, 212, 60), QColor(6, 182, 212, 200))

        if self.to_square:
            rect_to = self.square_to_rect(self.to_square)
            if rect_to:
                self._draw_glow_square(painter, rect_to, QColor(34, 197, 94, 60), QColor(34, 197, 94, 220))

        # 3. Glowing Neon Arrow
        if self.show_arrows and self.from_square and self.to_square:
            p_from = self.square_to_center_point(self.from_square)
            p_to = self.square_to_center_point(self.to_square)
            if p_from and p_to:
                self._draw_neon_arrow(painter, p_from, p_to, QColor(34, 197, 94))

        # 4. Vertical Eval Bar
        if self.show_eval_bar:
            bx, by, bw, bh = self.board_bbox
            self._draw_eval_bar(painter, bx - 34, by, 22, bh)

        # 5. Floating Glassmorphism HUD Badge
        if self.show_hud and self.best_move_san:
            bx, by, bw, bh = self.board_bbox
            self._draw_hud_badge(painter, bx, by - 62, bw, 50)

        painter.end()

    def _draw_glow_square(self, painter: QPainter, rect: QRectF, fill_col: QColor, border_col: QColor):
        """Draws a highlighted square with rounded corners and glowing border."""
        painter.save()
        radius = 8.0
        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(2, 2, -2, -2), radius, radius)

        # Fill
        painter.fillPath(path, QBrush(fill_col))

        # Glowing Border
        pen = QPen(border_col, 2.5)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.restore()

    def _draw_neon_arrow(self, painter: QPainter, p1: QPointF, p2: QPointF, base_color: QColor):
        """Draws multi-layer smooth arrow with neon glow."""
        painter.save()

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.hypot(dx, dy)
        if length < 10:
            painter.restore()
            return

        angle = math.atan2(dy, dx)
        head_length = min(36.0, length * 0.35)
        head_width = head_length * 0.75

        # Shorten shaft slightly so tip lands clean
        p2_short = QPointF(p2.x() - math.cos(angle) * (head_length * 0.3),
                           p2.y() - math.sin(angle) * (head_length * 0.3))

        # Outer Halo Glow
        glow_pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), 55), 18.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        painter.drawLine(p1, p2_short)

        # Mid Shaft
        mid_pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), 160), 10.0)
        mid_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(mid_pen)
        painter.drawLine(p1, p2_short)

        # Core Solid Shaft
        core_pen = QPen(QColor(255, 255, 255, 240), 4.5)
        core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(core_pen)
        painter.drawLine(p1, p2_short)

        # Arrow Head Triangle
        tip = p2
        left_corner = QPointF(
            p2.x() - head_length * math.cos(angle) + head_width * math.sin(angle),
            p2.y() - head_length * math.sin(angle) - head_width * math.cos(angle)
        )
        right_corner = QPointF(
            p2.x() - head_length * math.cos(angle) - head_width * math.sin(angle),
            p2.y() - head_length * math.sin(angle) + head_width * math.cos(angle)
        )

        arrow_poly = QPolygonF([tip, left_corner, right_corner])

        # Draw head glow
        painter.setPen(QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), 100), 4.0))
        painter.setBrush(QBrush(base_color))
        painter.drawPolygon(arrow_poly)

        painter.restore()

    def _draw_eval_bar(self, painter: QPainter, x: float, y: float, w: float, h: float):
        """Draws vertical evaluation bar matching Chess.com style."""
        painter.save()
        radius = 6.0
        rect = QRectF(x, y, w, h)

        # Background track (Dark)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        painter.fillPath(path, QBrush(QColor(38, 38, 44)))

        # White score percentage fill (0.0 to 1.0)
        # 50% is 0.0 eval. +5.0 eval is ~90%, -5.0 is ~10%
        white_pct = self.win_chance / 100.0
        white_h = h * white_pct
        white_y = y + (h - white_h)

        white_rect = QRectF(x, white_y, w, white_h)

        # Clip fill to rounded bar
        painter.setClipPath(path)
        painter.fillRect(white_rect, QColor(245, 245, 247))

        # Draw border
        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1.5))
        painter.drawPath(path)

        # Text score
        painter.setPen(QColor(18, 18, 24) if white_pct > 0.6 else QColor(240, 240, 245))
        font = QFont("Arial", 9, QFont.Weight.Bold)
        painter.setFont(font)

        eval_text = f"{self.eval_score:+.1f}" if self.eval_score is not None else "0.0"
        if self.is_mate:
            eval_text = f"M{self.mate_in}" if self.mate_in else "M"

        text_y = y + h - 8 if white_pct > 0.5 else y + 16
        painter.drawText(QRectF(x - 10, text_y - 8, w + 20, 16), Qt.AlignmentFlag.AlignCenter, eval_text)

        painter.restore()

    def _draw_hud_badge(self, painter: QPainter, x: float, y: float, w: float, h: float):
        """Draws floating glassmorphism HUD badge above board."""
        painter.save()
        badge_w = min(w, 420.0)
        badge_x = x + (w - badge_w) / 2.0
        rect = QRectF(badge_x, y, badge_w, h)

        radius = 12.0
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)

        # Glass background gradient
        grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        grad.setColorAt(0.0, QColor(24, 24, 30, 225))
        grad.setColorAt(1.0, QColor(14, 14, 18, 235))
        painter.fillPath(path, QBrush(grad))

        # Thin sleek border
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1.2))
        painter.drawPath(path)

        # Text labels
        font_label = QFont("Arial", 9, QFont.Weight.Normal)
        font_val = QFont("Arial", 14, QFont.Weight.Bold)

        # Best Move
        painter.setFont(font_label)
        painter.setPen(QColor(161, 161, 170))
        painter.drawText(QPointF(badge_x + 14, y + 18), "BEST MOVE")

        painter.setFont(font_val)
        painter.setPen(QColor(34, 197, 94))
        move_str = f"{self.best_move_san} ({self.from_square}→{self.to_square})"
        painter.drawText(QPointF(badge_x + 14, y + 38), move_str)

        # Win Chance / Status on Right
        painter.setFont(font_label)
        painter.setPen(QColor(161, 161, 170))
        painter.drawText(QRectF(badge_x + badge_w - 120, y + 8, 106, 14), Qt.AlignmentFlag.AlignRight, "WIN PROBABILITY")

        painter.setFont(font_val)
        painter.setPen(QColor(56, 189, 248))
        win_str = f"{self.win_chance:.0f}%"
        painter.drawText(QRectF(badge_x + badge_w - 120, y + 24, 106, 20), Qt.AlignmentFlag.AlignRight, win_str)

        painter.restore()

    def render_to_pixmap(self, width: int = 1280, height: int = 900) -> QPixmap:
        """Renders overlay onto an offscreen QPixmap for testing or export."""
        self.resize(width, height)
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        self.render(pixmap)
        return pixmap
