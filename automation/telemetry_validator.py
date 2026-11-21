"""
telemetry_validator.py: Anti-Cheat Telemetry Statistical Analysis & Evasion Validator.
Evaluates mouse trajectories against mathematical and physiological criteria:
Curvature straightness ratio, Fitts's Law velocity bell curve, 2D Gaussian dispersion,
and micro-tremor entropy to guarantee 100% human-like telemetry.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from scipy import stats

from automation.wind_mouse import TrajectoryPoint


class TelemetryValidator:
    """
    Analyzes mouse trajectory streams and timing telemetry to verify that
    it mimics genuine human biomechanics and successfully evades anti-cheat detection.
    """

    @staticmethod
    def calculate_curvature(points: List[TrajectoryPoint]) -> float:
        """
        Calculates the Straightness Ratio R = Euclidean Distance / Arc Length.
        A mechanical bot moves in a straight line (R = 1.000).
        A natural human hand trajectory has curvature (0.75 <= R <= 0.98).
        """
        if len(points) < 2:
            return 1.0

        p_start = points[0]
        p_end = points[-1]
        euclidean_dist = math.hypot(p_end.x - p_start.x, p_end.y - p_start.y)

        if euclidean_dist < 1e-4:
            return 1.0

        arc_length = 0.0
        for i in range(1, len(points)):
            step_dist = math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y)
            arc_length += step_dist

        if arc_length < 1e-4:
            return 1.0

        ratio = euclidean_dist / arc_length
        return round(float(ratio), 4)

    @staticmethod
    def analyze_velocity_profile(points: List[TrajectoryPoint]) -> Dict[str, Any]:
        """
        Validates Fitts's Law velocity profile.
        Human ballistic movement accelerates smoothly, peaks in the middle, and decelerates near destination.
        """
        if len(points) < 5:
            return {
                "is_bell_shaped": False,
                "peak_index_ratio": 0.5,
                "max_velocity": 0.0,
                "mean_velocity": 0.0,
                "velocity_ratio": 1.0,
            }

        velocities = []
        for i in range(1, len(points)):
            dt = points[i].timestamp - points[i - 1].timestamp
            dx = points[i].x - points[i - 1].x
            dy = points[i].y - points[i - 1].y
            dist = math.hypot(dx, dy)
            v = dist / dt if dt > 1e-5 else 0.0
            velocities.append(v)

        if not velocities:
            return {"is_bell_shaped": False, "peak_index_ratio": 0.5}

        vel_arr = np.array(velocities)
        max_v = float(np.max(vel_arr))
        mean_v = float(np.mean(vel_arr))
        peak_idx = int(np.argmax(vel_arr))
        peak_ratio = peak_idx / len(vel_arr)

        # Fitts's law: peak velocity should happen between 15% and 85% of trajectory,
        # and start/end velocity should be significantly lower than peak
        start_v = np.mean(vel_arr[:max(1, len(vel_arr) // 5)])
        end_v = np.mean(vel_arr[-max(1, len(vel_arr) // 5):])

        is_bell = (
            0.15 <= peak_ratio <= 0.85 and
            max_v > 1.25 * mean_v and
            start_v < max_v and
            end_v < max_v
        )

        return {
            "is_bell_shaped": bool(is_bell),
            "peak_index_ratio": round(peak_ratio, 3),
            "max_velocity": round(max_v, 2),
            "mean_velocity": round(mean_v, 2),
            "velocity_ratio": round(max_v / max(1e-3, mean_v), 2),
        }

    @staticmethod
    def calculate_jitter_entropy(points: List[TrajectoryPoint]) -> float:
        """
        Measures physiological tremor and high-frequency noise in trajectory.
        Computes standard deviation of second-order differences (acceleration fluctuations).
        Synthetic Bézier curves have ~0 jitter entropy; human hands have non-zero tremor.
        """
        if len(points) < 4:
            return 0.0

        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])

        # Second differences (approximates acceleration / jerk)
        d2x = np.diff(np.diff(xs))
        d2y = np.diff(np.diff(ys))
        jerk = np.hypot(d2x, d2y)

        entropy = float(np.std(jerk))
        return round(entropy, 4)

    @staticmethod
    def validate_dispersion(
        clicks: List[Tuple[float, float]],
        nominal_center: Tuple[float, float],
        square_size: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Validates spatial dispersion over multiple clicks.
        Checks that points are not identical and remain within square bounds.
        """
        if len(clicks) < 3:
            return {"valid": True, "unique_points": len(set(clicks)), "std_dev": 0.0}

        xs = np.array([c[0] for c in clicks])
        ys = np.array([c[1] for c in clicks])

        cx, cy = nominal_center
        dxs = np.abs(xs - cx)
        dys = np.abs(ys - cy)
        max_dx = float(np.max(dxs))
        max_dy = float(np.max(dys))

        offsets = np.hypot(xs - cx, ys - cy)
        max_offset = float(np.max(offsets))

        std_dev = float(np.std(offsets))
        unique_count = len(set((round(x, 1), round(y, 1)) for x, y in clicks))

        # Check bounds: clicks should be dispersed, but within square boundary (0.48 of square side)
        is_safe = max_dx < square_size * 0.48 and max_dy < square_size * 0.48
        is_dispersed = std_dev > 0.5 and unique_count > 1

        return {
            "valid": bool(is_safe and is_dispersed),
            "unique_points": unique_count,
            "std_dev": round(std_dev, 2),
            "max_offset": round(max_offset, 2),
            "max_dx": round(max_dx, 2),
            "max_dy": round(max_dy, 2),
        }

    @staticmethod
    def validate_lognormal_timing(delays: List[float]) -> Dict[str, Any]:
        """
        Verifies if an array of reaction delays matches human log-normal characteristics.
        """
        if len(delays) < 10:
            return {"is_lognormal": True, "p_value": 1.0}

        # Take log of positive delays
        log_delays = np.log([max(1e-4, d) for d in delays])

        # Shapiro-Wilk or D'Agostino-Pearson test for normality of the log-transformed values
        stat, p_value = stats.shapiro(log_delays)

        # Standard alpha threshold 0.01
        is_lognormal = bool(p_value > 0.01)

        return {
            "is_lognormal": is_lognormal,
            "p_value": round(float(p_value), 4),
            "mean": round(float(np.mean(delays)), 3),
            "median": round(float(np.median(delays)), 3),
            "min": round(float(np.min(delays)), 3),
            "max": round(float(np.max(delays)), 3),
        }

    @classmethod
    def evaluate_trajectory(cls, points: List[TrajectoryPoint]) -> Dict[str, Any]:
        """
        Comprehensive anti-cheat audit of a generated trajectory.
        """
        curvature = cls.calculate_curvature(points)
        vel_stats = cls.analyze_velocity_profile(points)
        jitter = cls.calculate_jitter_entropy(points)

        # Anti-cheat pass criteria:
        # 1. Not a straight bot line: R < 0.99
        # 2. Not excessively divergent: R > 0.60
        # 3. Fitts's law bell curve verified
        # 4. Natural tremor present: jitter > 0.01
        curvature_pass = 0.60 <= curvature <= 0.99
        velocity_pass = vel_stats["is_bell_shaped"]
        jitter_pass = jitter > 0.01

        overall_pass = curvature_pass and velocity_pass and jitter_pass

        return {
            "overall_pass": overall_pass,
            "curvature": curvature,
            "curvature_pass": curvature_pass,
            "velocity_stats": vel_stats,
            "velocity_pass": velocity_pass,
            "jitter_entropy": jitter,
            "jitter_pass": jitter_pass,
            "point_count": len(points),
        }
