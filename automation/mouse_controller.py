"""
mouse_controller.py: Human-like Mouse Automation Controller.
Supports Biomechanical WindMouse execution, Mock Telemetry recording for deterministic CI/E2E,
and Native macOS CoreGraphics event dispatching without external dependencies.
"""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import time
import math
import random
import ctypes
from ctypes import c_void_p, c_int, c_double, Structure

from automation.wind_mouse import TrajectoryPoint, generate_human_move_path, wind_mouse_trajectory
from automation.human_behavior import HumanThinkTimer, GaussianSquareSampler


class AbstractMouseController(ABC):
    """Abstract interface for mouse automation dispatchers."""

    @abstractmethod
    def get_position(self) -> Tuple[float, float]:
        """Returns current cursor position (x, y)."""
        pass

    @abstractmethod
    def move_to(self, x: float, y: float) -> None:
        """Instantly teleports cursor to (x, y). Used only internally by trajectory streamer."""
        pass

    @abstractmethod
    def mouse_down(self, button: str = "left") -> None:
        """Depresses given mouse button."""
        pass

    @abstractmethod
    def mouse_up(self, button: str = "left") -> None:
        """Releases given mouse button."""
        pass

    @abstractmethod
    def click(self, x: Optional[float] = None, y: Optional[float] = None, button: str = "left", duration: float = 0.05) -> None:
        """Performs a human click with down/up hold duration."""
        pass

    @abstractmethod
    def execute_trajectory(
        self,
        points: List[TrajectoryPoint],
        is_drag: bool = False,
        fast_mode: bool = False,
    ) -> None:
        """Streams a list of trajectory points to hardware or mock buffer."""
        pass


class MockMouseController(AbstractMouseController):
    """
    Deterministic In-Memory Mock Controller.
    Records every coordinate, timestamp, and event for anti-cheat statistical analysis
    and headless CI environments without requiring OS display or accessibility permissions.
    """

    def __init__(self, initial_x: float = 400.0, initial_y: float = 400.0, fast_mode: bool = True):
        self.current_x = float(initial_x)
        self.current_y = float(initial_y)
        self.current_time = 0.0
        self.fast_mode = fast_mode
        self.is_down = False
        self.recorded_points: List[TrajectoryPoint] = []
        self.recorded_events: List[Dict[str, Any]] = []

    def get_position(self) -> Tuple[float, float]:
        return self.current_x, self.current_y

    def move_to(self, x: float, y: float) -> None:
        self.current_x = float(x)
        self.current_y = float(y)
        pt = TrajectoryPoint(x=self.current_x, y=self.current_y, timestamp=round(self.current_time, 4))
        self.recorded_points.append(pt)
        self.recorded_events.append({"type": "move", "x": self.current_x, "y": self.current_y, "time": self.current_time})

    def mouse_down(self, button: str = "left") -> None:
        self.is_down = True
        self.recorded_events.append({"type": "down", "button": button, "x": self.current_x, "y": self.current_y, "time": self.current_time})

    def mouse_up(self, button: str = "left") -> None:
        self.is_down = False
        self.recorded_events.append({"type": "up", "button": button, "x": self.current_x, "y": self.current_y, "time": self.current_time})

    def click(self, x: Optional[float] = None, y: Optional[float] = None, button: str = "left", duration: float = 0.05) -> None:
        if x is not None and y is not None:
            self.move_to(x, y)
        self.mouse_down(button)
        self.current_time += duration
        if not self.fast_mode:
            time.sleep(duration)
        self.mouse_up(button)
        self.recorded_events.append({"type": "click", "button": button, "x": self.current_x, "y": self.current_y, "duration": duration, "time": self.current_time})

    def execute_trajectory(
        self,
        points: List[TrajectoryPoint],
        is_drag: bool = False,
        fast_mode: bool = False,
    ) -> None:
        if not points:
            return

        effective_fast_mode = fast_mode or self.fast_mode
        start_t = points[0].timestamp

        for i, pt in enumerate(points):
            self.current_x = pt.x
            self.current_y = pt.y
            self.current_time += (pt.timestamp - (points[i - 1].timestamp if i > 0 else start_t))
            self.recorded_points.append(TrajectoryPoint(x=pt.x, y=pt.y, timestamp=round(self.current_time, 4), velocity=pt.velocity))
            
            if not effective_fast_mode and i > 0:
                dt = pt.timestamp - points[i - 1].timestamp
                if dt > 0:
                    time.sleep(dt)

        self.recorded_events.append({
            "type": "drag_trajectory" if is_drag else "move_trajectory",
            "point_count": len(points),
            "start": (points[0].x, points[0].y),
            "end": (points[-1].x, points[-1].y),
            "time": self.current_time,
        })


class CGPoint(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class NativeMacMouseController(AbstractMouseController):
    """
    Direct macOS CoreGraphics / ApplicationServices C-Types Mouse Dispatcher.
    High performance, zero-latency, and requires no external third-party compilation.
    """

    kCGEventLeftMouseDown = 1
    kCGEventLeftMouseUp = 2
    kCGEventRightMouseDown = 3
    kCGEventRightMouseUp = 4
    kCGEventMouseMoved = 5
    kCGEventLeftMouseDragged = 6
    kCGHIDEventTap = 0

    def __init__(self):
        try:
            self.cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
            self.cg.CGEventCreateMouseEvent.argtypes = [c_void_p, c_int, CGPoint, c_int]
            self.cg.CGEventCreateMouseEvent.restype = c_void_p
            self.cg.CGEventPost.argtypes = [c_int, c_void_p]
            self.cg.CGEventPost.restype = None
            self.cg.CFRelease.argtypes = [c_void_p]
            self.cg.CFRelease.restype = None
            self.available = True
        except Exception:
            self.available = False

        self._curr_x = 500.0
        self._curr_y = 500.0

    def get_position(self) -> Tuple[float, float]:
        return self._curr_x, self._curr_y

    def _post_event(self, event_type: int, x: float, y: float, button: int = 0) -> None:
        if not self.available:
            return
        pt = CGPoint(float(x), float(y))
        evt = self.cg.CGEventCreateMouseEvent(None, event_type, pt, button)
        if evt:
            self.cg.CGEventPost(self.kCGHIDEventTap, evt)
            self.cg.CFRelease(evt)
        self._curr_x = float(x)
        self._curr_y = float(y)

    def move_to(self, x: float, y: float) -> None:
        self._post_event(self.kCGEventMouseMoved, x, y)

    def mouse_down(self, button: str = "left") -> None:
        evt_type = self.kCGEventLeftMouseDown if button == "left" else self.kCGEventRightMouseDown
        self._post_event(evt_type, self._curr_x, self._curr_y, 0 if button == "left" else 1)

    def mouse_up(self, button: str = "left") -> None:
        evt_type = self.kCGEventLeftMouseUp if button == "left" else self.kCGEventRightMouseUp
        self._post_event(evt_type, self._curr_x, self._curr_y, 0 if button == "left" else 1)

    def click(self, x: Optional[float] = None, y: Optional[float] = None, button: str = "left", duration: float = 0.06) -> None:
        if x is not None and y is not None:
            self.move_to(x, y)
            time.sleep(0.03)
        self.mouse_down(button)
        time.sleep(duration)
        self.mouse_up(button)
        time.sleep(0.02)

    def execute_trajectory(
        self,
        points: List[TrajectoryPoint],
        is_drag: bool = False,
        fast_mode: bool = False,
    ) -> None:
        if not points:
            return

        evt_type = self.kCGEventLeftMouseDragged if is_drag else self.kCGEventMouseMoved

        for i, pt in enumerate(points):
            self._post_event(evt_type, pt.x, pt.y)
            if not fast_mode and i > 0:
                dt = pt.timestamp - points[i - 1].timestamp
                if dt > 0:
                    time.sleep(dt)


class HumanMouseController:
    """
    High-Level Human-Like Mouse Automation Agent.
    Coordinates WindMouse biomechanical trajectories, human cognitive thinking pauses,
    and 2D Gaussian chess square dispersion to completely evade anti-cheat heuristics.
    """

    def __init__(
        self,
        backend: Optional[AbstractMouseController] = None,
        prefer_drag: bool = False,
        fast_mode: bool = False,
    ):
        """
        Args:
            backend: Underlying AbstractMouseController (defaults to NativeMac if available, else Mock).
            prefer_drag: If True, uses Drag & Drop instead of Click-Click.
            fast_mode: If True, skips actual sleep pauses (for unit tests / benchmarks).
        """
        if backend is not None:
            self.backend = backend
        else:
            native = NativeMacMouseController()
            self.backend = native if native.available else MockMouseController(fast_mode=fast_mode)

        self.prefer_drag = prefer_drag
        self.fast_mode = fast_mode
        self.timer = HumanThinkTimer()
        self.sampler = GaussianSquareSampler()

    def play_move(
        self,
        from_square: str,
        to_square: str,
        board_bbox: Tuple[int, int, int, int],
        orientation_white_bottom: bool = True,
        is_drag: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete human chess move (e.g. 'e2' -> 'e4') on screen.
        
        Args:
            from_square: Starting square ('e2').
            to_square: Destination square ('e4').
            board_bbox: (x, y, width, height) pixel bounding box of board.
            orientation_white_bottom: Board perspective.
            is_drag: Force drag mode, or defaults to self.prefer_drag.
            context: Optional tactical context for cognitive timer (e.g. is_in_check, is_recapture).
            
        Returns:
            Dictionary with move telemetry execution summary.
        """
        ctx = context or {}
        use_drag = self.prefer_drag if is_drag is None else is_drag

        # 1. Human Cognitive Deliberation Delay
        think_time = self.timer.calculate_delay(
            is_opening_book=ctx.get("is_opening_book", False),
            is_recapture=ctx.get("is_recapture", False),
            is_in_check=ctx.get("is_in_check", False),
            is_blunder_check=ctx.get("is_blunder_check", False),
            time_trouble=ctx.get("time_trouble", False),
            ply_count=ctx.get("ply_count", 1),
        )

        if not self.fast_mode:
            time.sleep(think_time)

        # 2. Gaussian Square Sampling for start and destination
        start_x, start_y = self.sampler.sample_square_from_bbox(
            board_bbox, from_square, orientation_white_bottom
        )
        dest_x, dest_y = self.sampler.sample_square_from_bbox(
            board_bbox, to_square, orientation_white_bottom
        )

        curr_x, curr_y = self.backend.get_position()

        # 3. Trajectory 1: Move cursor to source square
        path_to_start = generate_human_move_path(
            curr_x, curr_y, start_x, start_y, enable_overshoot=True, enable_tremor=True
        )
        self.backend.execute_trajectory(path_to_start, is_drag=False, fast_mode=self.fast_mode)

        click_down_duration = self.timer.calculate_click_duration()

        total_points = len(path_to_start)

        if use_drag:
            # Mode A: Drag and Drop
            self.backend.mouse_down("left")
            if not self.fast_mode:
                time.sleep(click_down_duration)

            path_to_dest = generate_human_move_path(
                start_x, start_y, dest_x, dest_y, enable_overshoot=True, enable_tremor=True
            )
            self.backend.execute_trajectory(path_to_dest, is_drag=True, fast_mode=self.fast_mode)

            drag_pause = self.timer.calculate_drag_pause()
            if not self.fast_mode:
                time.sleep(drag_pause)
            self.backend.mouse_up("left")
            total_points += len(path_to_dest)
        else:
            # Mode B: Click-Click
            self.backend.click(duration=click_down_duration)
            time.sleep(random.uniform(0.06, 0.12))

            path_to_dest = generate_human_move_path(
                start_x, start_y, dest_x, dest_y, enable_overshoot=True, enable_tremor=True
            )
            self.backend.execute_trajectory(path_to_dest, is_drag=False, fast_mode=self.fast_mode)

            time.sleep(0.04)
            dest_click_duration = self.timer.calculate_click_duration()
            self.backend.click(duration=dest_click_duration)
            total_points += len(path_to_dest)

        return {
            "move": f"{from_square}{to_square}",
            "mode": "drag" if use_drag else "click_click",
            "from_coords": (start_x, start_y),
            "to_coords": (dest_x, dest_y),
            "think_time_sec": think_time,
            "total_trajectory_points": total_points,
            "path_to_dest_points": len(path_to_dest),
            "success": True,
        }
