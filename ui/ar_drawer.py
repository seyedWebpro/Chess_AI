"""
ARDrawer: Augmented Reality (AR) visual rendering pipeline for Chess AI.
Features:
- Semi-transparent neon arrows with glow and shadows.
- Polygon square highlights (unwarped to camera perspective).
- Vertical Chess.com-style evaluation bar (Eval Bar).
- Glassmorphism HUD info badge (Best Move, Score, Depth).
- Sub-3ms rendering speed on 1080p/720p frames.
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import cv2
import numpy as np
import chess
from loguru import logger
from vision.calibrator import BoardCalibrator


class ARDrawer:
    """
    Renders AR overlays on raw camera frames or normalized 800x800 board views.
    """

    # BGR Color Palette (Modern Dark / Neon Theme)
    COLOR_NEON_GREEN = (94, 197, 34)    # Best move (#22c55e)
    COLOR_AMBER_YELLOW = (8, 179, 234)  # Defense / Alternative (#eab308)
    COLOR_CRIMSON_RED = (68, 68, 239)   # Threat / Check (#ef4444)
    COLOR_CYAN_BLUE = (235, 176, 56)    # Special (#38bdf8)
    COLOR_DARK_PANEL = (27, 24, 24)     # Semi-transparent HUD background
    COLOR_TEXT_WHITE = (250, 250, 250)
    COLOR_TEXT_MUTED = (160, 160, 161)

    def __init__(self, enable_glow: bool = True):
        self.enable_glow = enable_glow

    def draw_smooth_arrow(
        self,
        frame: np.ndarray,
        pt_start: Tuple[int, int],
        pt_end: Tuple[int, int],
        color: Tuple[int, int, int] = COLOR_NEON_GREEN,
        thickness: int = 6,
        tip_length: float = 0.25,
    ) -> np.ndarray:
        """
        Renders a thick, modern arrow with semi-transparent outer glow and shadow.
        Optimized with bounding box ROI blending for <0.5ms execution.
        """
        h_frame, w_frame = frame.shape[:2]

        # Outer glow layer using ROI blending
        if self.enable_glow:
            glow_pad = thickness + 16
            x_min = max(0, min(pt_start[0], pt_end[0]) - glow_pad)
            x_max = min(w_frame, max(pt_start[0], pt_end[0]) + glow_pad)
            y_min = max(0, min(pt_start[1], pt_end[1]) - glow_pad)
            y_max = min(h_frame, max(pt_start[1], pt_end[1]) + glow_pad)

            if x_max > x_min and y_max > y_min:
                roi = frame[y_min:y_max, x_min:x_max]
                glow_roi = roi.copy()
                pt1_roi = (pt_start[0] - x_min, pt_start[1] - y_min)
                pt2_roi = (pt_end[0] - x_min, pt_end[1] - y_min)

                cv2.arrowedLine(
                    glow_roi,
                    pt1_roi,
                    pt2_roi,
                    color,
                    thickness=thickness + 8,
                    tipLength=tip_length,
                    line_type=cv2.LINE_AA,
                )
                cv2.addWeighted(glow_roi, 0.35, roi, 0.65, 0, roi)

        # Core sharp vector line
        cv2.arrowedLine(
            frame,
            pt_start,
            pt_end,
            color,
            thickness=thickness,
            tipLength=tip_length,
            line_type=cv2.LINE_AA,
        )

        # Circle at base point
        cv2.circle(frame, pt_start, thickness + 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, pt_start, thickness - 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return frame

    def draw_square_polygon(
        self,
        frame: np.ndarray,
        pts_quad: np.ndarray,
        color: Tuple[int, int, int] = COLOR_NEON_GREEN,
        alpha: float = 0.35,
        border: bool = True,
    ) -> np.ndarray:
        """
        Draws a semi-transparent filled polygon over a square on a tilted camera frame.
        Uses ROI bounding-box blending for sub-millisecond execution.
        pts_quad: Array of 4 corner points (shape: (4, 2)).
        """
        h_frame, w_frame = frame.shape[:2]
        cnt = pts_quad.astype(np.int32).reshape((-1, 1, 2))

        # Compute bounding rectangle for fast ROI blending
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(w_frame, x + w)
        y_max = min(h_frame, y + h)

        if x_max > x_min and y_max > y_min:
            roi = frame[y_min:y_max, x_min:x_max]
            roi_overlay = roi.copy()

            # Shift contour relative to ROI
            cnt_shifted = cnt.copy()
            cnt_shifted[:, :, 0] -= x_min
            cnt_shifted[:, :, 1] -= y_min

            cv2.fillPoly(roi_overlay, [cnt_shifted], color, lineType=cv2.LINE_AA)
            cv2.addWeighted(roi_overlay, alpha, roi, 1.0 - alpha, 0, roi)

            if border:
                cv2.polylines(roi, [cnt_shifted], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        return frame

        return frame

    def draw_eval_bar(
        self,
        frame: np.ndarray,
        win_chance_pct: float,
        eval_str: str,
        x: int = 30,
        y: int = 80,
        w: int = 30,
        h: int = 560,
    ) -> np.ndarray:
        """
        Draws a vertical Chess.com-style evaluation bar on the side of the frame.
        White win chance grows from bottom to top.
        """
        # Outer panel
        cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (10, 10, 10), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)

        # White fill height (from bottom)
        fill_h = int(np.clip(win_chance_pct / 100.0 * h, 0, h))
        white_top_y = y + h - fill_h

        # White section
        cv2.rectangle(frame, (x, white_top_y), (x + w, y + h), (245, 245, 245), -1)

        # Subtle border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1, lineType=cv2.LINE_AA)

        # Center line at 50%
        mid_y = y + h // 2
        cv2.line(frame, (x - 4, mid_y), (x + w + 4, mid_y), (140, 140, 140), 1)

        # Evaluation label at bottom or top depending on advantage
        label_y = y + 25 if win_chance_pct < 50 else y + h - 12
        txt_color = (250, 250, 250) if win_chance_pct < 50 else (20, 20, 20)

        # Shorten text
        short_eval = eval_str.replace(" ", "")
        cv2.putText(
            frame,
            short_eval,
            (x - 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            txt_color,
            1,
            lineType=cv2.LINE_AA,
        )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        analysis: Dict[str, Any],
        x: int = 80,
        y: int = 30,
        w: int = 380,
        h: int = 65,
    ) -> np.ndarray:
        """
        Draws a sleek floating glassmorphism HUD card with real-time AI decision info.
        Uses ROI blending for high FPS.
        """
        h_frame, w_frame = frame.shape[:2]
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(w_frame, x + w)
        y_max = min(h_frame, y + h)

        if x_max > x_min and y_max > y_min:
            roi = frame[y_min:y_max, x_min:x_max]
            roi_overlay = np.full_like(roi, self.COLOR_DARK_PANEL)
            cv2.addWeighted(roi_overlay, 0.75, roi, 0.25, 0, roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 75), 1, lineType=cv2.LINE_AA)

        # Left accent stripe (Green for winning, yellow for equal, red for losing)
        win_pct = analysis.get("win_chance_pct", 50.0)
        stripe_color = (
            self.COLOR_NEON_GREEN if win_pct >= 55.0
            else self.COLOR_AMBER_YELLOW if win_pct >= 45.0
            else self.COLOR_CRIMSON_RED
        )
        cv2.rectangle(frame, (x, y), (x + 6, y + h), stripe_color, -1)

        # Line 1: Best Move
        best_san = analysis.get("best_move_san", "N/A")
        best_uci = analysis.get("best_move", "")
        txt_line1 = f"AI Best: {best_san} ({best_uci})"
        cv2.putText(frame, txt_line1, (x + 18, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT_WHITE, 2, cv2.LINE_AA)

        # Line 2: Eval & Win Chance
        eval_str = analysis.get("eval_str", "0.0")
        depth = analysis.get("depth_reached", 0)
        txt_line2 = f"Eval: {eval_str}  |  Win: {win_pct}%  |  Depth: {depth}"
        cv2.putText(frame, txt_line2, (x + 18, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        return frame

    def render_ar_on_camera(
        self,
        camera_frame: np.ndarray,
        analysis: Dict[str, Any],
        calibrator: BoardCalibrator,
        check_square: Optional[str] = None,
    ) -> np.ndarray:
        """
        High-level AR pipeline: Unwarps chess coordinates onto raw camera feed and renders all visuals.
        
        Args:
            camera_frame: Raw camera frame (e.g. 1280x720).
            analysis: Dict from EngineManager.evaluate_position().
            calibrator: Calibrator containing forward and inverse homography.
            check_square: Square name of King under check (e.g. 'e1' or 'e8').
            
        Returns:
            AR-annotated camera frame.
        """
        output_frame = camera_frame.copy()

        from_sq = analysis.get("from_square")
        to_sq = analysis.get("to_square")

        # 1. Highlight Squares
        if from_sq and to_sq:
            # Unwarp 4 corners of from_square and to_square
            pts_from = self._unwarp_square_corners(from_sq, calibrator)
            pts_to = self._unwarp_square_corners(to_sq, calibrator)

            # Draw semi-transparent square overlays
            self.draw_square_polygon(output_frame, pts_from, color=self.COLOR_AMBER_YELLOW, alpha=0.3)
            self.draw_square_polygon(output_frame, pts_to, color=self.COLOR_NEON_GREEN, alpha=0.45)

        # Check square highlight
        if check_square:
            pts_check = self._unwarp_square_corners(check_square, calibrator)
            self.draw_square_polygon(output_frame, pts_check, color=self.COLOR_CRIMSON_RED, alpha=0.5)

        # 2. Draw Vector Arrow
        if from_sq and to_sq:
            if hasattr(calibrator, "get_unwarped_center"):
                cam_start = calibrator.get_unwarped_center(from_sq)
                cam_end = calibrator.get_unwarped_center(to_sq)
            else:
                cx_from, cy_from = calibrator.get_square_center(from_sq)
                cx_to, cy_to = calibrator.get_square_center(to_sq)
                cam_start = calibrator.unwarp_point(cx_from, cy_from)
                cam_end = calibrator.unwarp_point(cx_to, cy_to)

            self.draw_smooth_arrow(
                output_frame,
                cam_start,
                cam_end,
                color=self.COLOR_NEON_GREEN,
                thickness=6,
                tip_length=0.22,
            )

        # 3. Draw Vertical Eval Bar
        win_pct = analysis.get("win_chance_pct", 50.0)
        eval_str = analysis.get("eval_str", "0.0")
        self.draw_eval_bar(output_frame, win_chance_pct=win_pct, eval_str=eval_str)

        # 4. Draw Floating HUD
        self.draw_hud(output_frame, analysis)

        return output_frame

    def render_ar_on_warped(
        self,
        warped_800: np.ndarray,
        analysis: Dict[str, Any],
        calibrator: BoardCalibrator,
    ) -> np.ndarray:
        """
        Renders AR elements directly onto the 800x800 normalized board view.
        """
        output = warped_800.copy()
        from_sq = analysis.get("from_square")
        to_sq = analysis.get("to_square")

        if from_sq and to_sq:
            # Highlight from and to
            x1_f, y1_f, x2_f, y2_f = calibrator.get_square_rect(from_sq)
            x1_t, y1_t, x2_t, y2_t = calibrator.get_square_rect(to_sq)

            quad_from = np.array([[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]], dtype=np.float32)
            quad_to = np.array([[x1_t, y1_t], [x2_t, y1_t], [x2_t, y2_t], [x1_t, y2_t]], dtype=np.float32)

            self.draw_square_polygon(output, quad_from, color=self.COLOR_AMBER_YELLOW, alpha=0.35)
            self.draw_square_polygon(output, quad_to, color=self.COLOR_NEON_GREEN, alpha=0.45)

            # Vector arrow
            pt_start = calibrator.get_square_center(from_sq)
            pt_end = calibrator.get_square_center(to_sq)

            self.draw_smooth_arrow(output, pt_start, pt_end, color=self.COLOR_NEON_GREEN, thickness=7)

        return output

    @staticmethod
    def _unwarp_square_corners(square_name: str, calibrator: BoardCalibrator) -> np.ndarray:
        """Helper to get 4 unwarped corner points of a square in camera frame."""
        if hasattr(calibrator, "get_unwarped_square"):
            return calibrator.get_unwarped_square(square_name)
        x1, y1, x2, y2 = calibrator.get_square_rect(square_name)
        corners_800 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        unwarped_pts = [calibrator.unwarp_point(x, y) for x, y in corners_800]
        return np.array(unwarped_pts, dtype=np.float32)
