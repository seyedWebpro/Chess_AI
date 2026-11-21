"""
automation package: Human-like mouse automation, WindMouse trajectory generation,
cognitive reaction timing, and anti-cheat telemetry validation.
"""

from automation.wind_mouse import (
    TrajectoryPoint,
    wind_mouse_trajectory,
    generate_human_move_path,
)
from automation.human_behavior import (
    HumanThinkTimer,
    GaussianSquareSampler,
)
from automation.mouse_controller import (
    AbstractMouseController,
    MockMouseController,
    NativeMacMouseController,
    HumanMouseController,
)
from automation.telemetry_validator import (
    TelemetryValidator,
)

__all__ = [
    "TrajectoryPoint",
    "wind_mouse_trajectory",
    "generate_human_move_path",
    "HumanThinkTimer",
    "GaussianSquareSampler",
    "AbstractMouseController",
    "MockMouseController",
    "NativeMacMouseController",
    "HumanMouseController",
    "TelemetryValidator",
]
