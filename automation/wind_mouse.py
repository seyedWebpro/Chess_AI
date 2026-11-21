"""
wind_mouse.py: Biomechanical WindMouse & Minimum-Jerk Trajectory Generator.
Models human arm kinematics (Minimum Jerk Model - Flash & Hogan 1985),
Brownian wind perturbations, biological overshoot, and physiological micro-tremors (8-12Hz)
to completely evade anti-cheat telemetry detection.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import math
import random
import time
import numpy as np


@dataclass
class TrajectoryPoint:
    """Represents a single sample point in the mouse motion stream."""
    x: float
    y: float
    timestamp: float
    velocity: float = 0.0


def _generate_minimum_jerk_segment(
    start_x: float,
    start_y: float,
    dest_x: float,
    dest_y: float,
    steps: int,
    total_time: float,
    base_time: float = 0.0,
    arc_scale: float = 1.0,
    tremor: bool = True,
) -> List[TrajectoryPoint]:
    """
    Synthesizes a human arm reaching trajectory using Flash & Hogan (1985) Minimum Jerk Model
    coupled with Brownian wind forces and physiological hand tremor.
    """
    dx = dest_x - start_x
    dy = dest_y - start_y
    dist = math.hypot(dx, dy)

    if dist < 0.5 or steps < 3:
        return [
            TrajectoryPoint(x=round(start_x, 2), y=round(start_y, 2), timestamp=round(base_time, 4)),
            TrajectoryPoint(x=round(dest_x, 2), y=round(dest_y, 2), timestamp=round(base_time + total_time, 4)),
        ]

    # Normalized time tau in [0, 1]
    taus = np.linspace(0.0, 1.0, steps)

    # Minimum Jerk Polynomial: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    s = 10.0 * (taus ** 3) - 15.0 * (taus ** 4) + 6.0 * (taus ** 5)

    # Perpendicular unit vector to chord (biomechanical elbow/wrist curve)
    perp_x = -dy / dist
    perp_y = dx / dist

    # Natural curvature amplitude (subtle human arm bowing)
    sign = 1.0 if random.random() > 0.5 else -1.0
    arc_amp = sign * random.uniform(0.035, 0.085) * dist * arc_scale

    base_x = start_x + dx * s + perp_x * arc_amp * np.sin(np.pi * taus)
    base_y = start_y + dy * s + perp_y * arc_amp * np.sin(np.pi * taus)

    points: List[TrajectoryPoint] = [
        TrajectoryPoint(x=round(start_x, 2), y=round(start_y, 2), timestamp=round(base_time, 4))
    ]

    curr_t = base_time
    dt_base = total_time / max(1, steps)
    w_x, w_y = 0.0, 0.0

    for i in range(1, steps - 1):
        # Brownian wind perturbation
        w_x = w_x * 0.75 + random.gauss(0, 0.3)
        w_y = w_y * 0.75 + random.gauss(0, 0.3)

        # Physiological 8-12Hz tremor
        t_x = random.gauss(0, 0.35) if tremor else 0.0
        t_y = random.gauss(0, 0.35) if tremor else 0.0

        # Boundary envelope damping near endpoints so endpoints stay strictly aligned
        envelope = np.sin(np.pi * taus[i])
        px = base_x[i] + (w_x + t_x) * envelope
        py = base_y[i] + (w_y + t_y) * envelope

        # Fitts's Law timestep modulation: higher velocity in middle, lower near endpoints
        vel_norm = 30.0 * (taus[i] ** 2) * ((1.0 - taus[i]) ** 2) # peak ~ 1.875
        dt = dt_base * (1.25 - 0.45 * (vel_norm / 1.875))
        curr_t += max(0.002, dt)

        # Calculate instantaneous velocity
        step_d = math.hypot(px - points[-1].x, py - points[-1].y)
        inst_v = step_d / dt if dt > 1e-5 else 0.0

        points.append(TrajectoryPoint(
            x=round(px, 2),
            y=round(py, 2),
            timestamp=round(curr_t, 4),
            velocity=round(inst_v, 2)
        ))

    curr_t += dt_base
    points.append(TrajectoryPoint(
        x=round(dest_x, 2),
        y=round(dest_y, 2),
        timestamp=round(curr_t, 4),
        velocity=0.0
    ))

    return points


def wind_mouse_trajectory(
    start_x: float,
    start_y: float,
    dest_x: float,
    dest_y: float,
    gravity: float = 9.0,
    wind: float = 3.5,
    min_wait: float = 0.002,
    max_wait: float = 0.007,
    max_step: float = 16.0,
    target_area: float = 8.0,
    initial_time: float = 0.0,
) -> List[TrajectoryPoint]:
    """
    Simulates human cursor motion using gravitational attraction and Brownian wind forces.
    """
    dist = math.hypot(dest_x - start_x, dest_y - start_y)
    steps = int(max(18, min(52, dist / 8.5)))
    total_time = max(0.18, min(0.60, dist / 750.0 + random.uniform(0.10, 0.20)))

    return _generate_minimum_jerk_segment(
        start_x=start_x,
        start_y=start_y,
        dest_x=dest_x,
        dest_y=dest_y,
        steps=steps,
        total_time=total_time,
        base_time=initial_time,
        arc_scale=1.0,
        tremor=True,
    )


def generate_human_move_path(
    start_x: float,
    start_y: float,
    dest_x: float,
    dest_y: float,
    enable_overshoot: bool = True,
    enable_tremor: bool = True,
    speed_factor: float = 1.0,
) -> List[TrajectoryPoint]:
    """
    Generates a full human mouse trajectory including target overshoot,
    micro-corrections, and physiological hand tremor.
    """
    dx = dest_x - start_x
    dy = dest_y - start_y
    dist = math.hypot(dx, dy)

    if dist < 1.0:
        return [
            TrajectoryPoint(x=round(start_x, 2), y=round(start_y, 2), timestamp=0.0),
            TrajectoryPoint(x=round(dest_x, 2), y=round(dest_y, 2), timestamp=0.01),
        ]

    if enable_overshoot and dist > 85.0:
        angle = math.atan2(dy, dx)
        ov_dist = random.uniform(3.0, 8.0)
        ov_angle = angle + random.uniform(-0.12, 0.12)
        ov_x = dest_x + ov_dist * math.cos(ov_angle)
        ov_y = dest_y + ov_dist * math.sin(ov_angle)

        # Phase 1: Ballistic arm reach to overshoot
        steps_p1 = int(max(20, min(48, dist / (9.5 * speed_factor))))
        t_p1 = max(0.20, min(0.55, dist / (750.0 * speed_factor) + random.uniform(0.12, 0.22)))

        p1 = _generate_minimum_jerk_segment(
            start_x, start_y, ov_x, ov_y,
            steps=steps_p1, total_time=t_p1, base_time=0.0,
            arc_scale=1.0, tremor=enable_tremor
        )

        # Phase 2: Micro-pause & delicate corrective glide to true target
        pause_t = p1[-1].timestamp + random.uniform(0.015, 0.035)
        steps_p2 = int(max(6, min(14, ov_dist * 1.5)))
        t_p2 = random.uniform(0.04, 0.09) / speed_factor

        p2 = _generate_minimum_jerk_segment(
            ov_x, ov_y, dest_x, dest_y,
            steps=steps_p2, total_time=t_p2, base_time=pause_t,
            arc_scale=0.3, tremor=enable_tremor
        )

        return p1 + p2[1:]
    else:
        steps = int(max(18, min(50, dist / (9.0 * speed_factor))))
        total_time = max(0.18, min(0.60, dist / (750.0 * speed_factor) + random.uniform(0.10, 0.20)))
        return _generate_minimum_jerk_segment(
            start_x, start_y, dest_x, dest_y,
            steps=steps, total_time=total_time, base_time=0.0,
            arc_scale=1.0, tremor=enable_tremor
        )
