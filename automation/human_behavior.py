"""
human_behavior.py: Biomechanical & Cognitive Human Behavior Modeling.
Simulates human cognitive delays (log-normal reaction time), opening book memory,
and 2D Gaussian spatial dispersion on chess board squares to evade anti-cheat heuristics.
"""

from typing import Tuple, Optional
import math
import random
import numpy as np


class HumanThinkTimer:
    """
    Simulates human cognitive reaction times based on psychological research:
    human decision-making latency follows a Log-Normal distribution.
    """

    def __init__(
        self,
        base_mu: float = 0.5,       # Mean of log-scale duration (~1.65 seconds)
        base_sigma: float = 0.35,   # Variance of log-scale duration
        min_delay: float = 0.25,    # Hard physical floor (fastest human reflexes)
        max_delay: float = 7.5,     # Tactical ponder cap
    ):
        self.base_mu = base_mu
        self.base_sigma = base_sigma
        self.min_delay = min_delay
        self.max_delay = max_delay

    def calculate_delay(
        self,
        is_opening_book: bool = False,
        is_recapture: bool = False,
        is_in_check: bool = False,
        is_blunder_check: bool = False,
        time_trouble: bool = False,
        ply_count: int = 1,
    ) -> float:
        """
        Calculates realistic human reaction delay conditioned on board context.
        
        Args:
            is_opening_book: True if move is played from memorized opening theory.
            is_recapture: Obvious immediate recapture of opponent's piece.
            is_in_check: King under attack requiring defensive deliberation.
            is_blunder_check: Complex tactical calculation phase.
            time_trouble: Clock low (e.g. blitz/bullet bullet scramble).
            ply_count: Game move number.
        
        Returns:
            Delay in seconds as a float.
        """
        mu = self.base_mu
        sigma = self.base_sigma

        if time_trouble:
            # Bullet scramble: extremely fast, low variance
            mu -= 1.1
            sigma *= 0.7
        elif is_opening_book and ply_count <= 14:
            # Memorized opening: fast, confident execution
            mu -= 0.65
            sigma *= 0.8
        elif is_recapture:
            # Obvious recapture: quick reaction
            mu -= 0.45
            sigma *= 0.85
        elif is_in_check:
            # Under check: deliberate care
            mu += 0.35
            sigma *= 1.1
        elif is_blunder_check:
            # Critical tactical moment: prolonged thinking
            mu += 0.60
            sigma *= 1.15

        # Sample from log-normal distribution
        raw_delay = float(np.random.lognormal(mu, sigma))

        # Clamp between physiological floor and max ponder cap
        clamped_delay = max(self.min_delay, min(self.max_delay, raw_delay))
        return round(clamped_delay, 3)

    def calculate_click_duration(self) -> float:
        """
        Duration a human keeps the mouse button pressed down before releasing.
        Typically 50ms to 120ms with log-normal distribution.
        """
        # Log-normal with median ~75ms
        press_time = float(np.random.lognormal(-2.6, 0.28))
        return round(max(0.040, min(0.180, press_time)), 3)

    def calculate_drag_pause(self) -> float:
        """
        Micro-pause at target square before mouse release during drag-and-drop.
        Typically 25ms to 75ms.
        """
        pause = float(np.random.lognormal(-3.3, 0.35))
        return round(max(0.020, min(0.110, pause)), 3)


class GaussianSquareSampler:
    """
    Samples human click/drag landing coordinates within a chess square.
    Real humans do not click the mathematical center (x_c, y_c) of a square;
    their clicks form a 2D Gaussian distribution centered near the square midpoint.
    """

    def __init__(
        self,
        std_ratio: float = 0.20,       # Standard deviation as fraction of square side
        max_deviation_ratio: float = 0.38, # Cutoff radius to prevent clicking adjacent squares
    ):
        self.std_ratio = std_ratio
        self.max_deviation_ratio = max_deviation_ratio

    def sample_point(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> Tuple[float, float]:
        """
        Samples a realistic human click coordinate within [x_min, y_min, x_max, y_max].
        
        Args:
            x_min, y_min: Top-left coordinate of the square.
            x_max, y_max: Bottom-right coordinate of the square.
            
        Returns:
            Tuple (x, y) rounded to 1 decimal place.
        """
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2.0
        center_y = y_min + height / 2.0

        sigma_x = width * self.std_ratio
        sigma_y = height * self.std_ratio
        max_dev_x = width * self.max_deviation_ratio
        max_dev_y = height * self.max_deviation_ratio

        # Truncated 2D Gaussian with radial bound
        dx = float(np.random.normal(0, sigma_x))
        dy = float(np.random.normal(0, sigma_y))

        rad = math.hypot(dx, dy)
        max_rad = min(width, height) * self.max_deviation_ratio
        if rad > max_rad and rad > 1e-5:
            dx = (dx / rad) * max_rad
            dy = (dy / rad) * max_rad

        return round(center_x + dx, 1), round(center_y + dy, 1)

    def sample_square_from_bbox(
        self,
        board_bbox: Tuple[int, int, int, int],
        square_name: str,
        orientation_white_bottom: bool = True,
    ) -> Tuple[float, float]:
        """
        Calculates screen coordinate for a square algebraic notation (e.g. 'e4').
        
        Args:
            board_bbox: (x, y, width, height) of entire chess board on screen.
            square_name: Algebraic notation 'a1'..'h8'.
            orientation_white_bottom: True if white pieces at bottom.
            
        Returns:
            Screen coordinate (x, y).
        """
        bx, by, bw, bh = board_bbox
        sq_w = bw / 8.0
        sq_h = bh / 8.0

        file_idx = ord(square_name[0].lower()) - ord('a') # 0..7 for a..h
        rank_idx = int(square_name[1]) - 1                # 0..7 for 1..8

        if orientation_white_bottom:
            col = file_idx
            row = 7 - rank_idx
        else:
            col = 7 - file_idx
            row = rank_idx

        x_min = bx + col * sq_w
        y_min = by + row * sq_h
        x_max = x_min + sq_w
        y_max = y_min + sq_h

        return self.sample_point(x_min, y_min, x_max, y_max)
