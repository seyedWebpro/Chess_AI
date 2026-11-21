"""
ScreenCapture: High-performance screen capture pipeline for online chess play.
Features:
- Fast C-based screen capture via mss.
- Region of Interest (ROI) bounding box grabbing.
- ThreadedScreenCapture for 0ms buffer lag in real-time loops.
- MockScreenCapture for CI and unit testing without a physical monitor.
"""

import time
import threading
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import cv2
from loguru import logger

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    logger.warning("mss is not installed. ScreenCapture will use fallback.")


class ScreenCapture:
    """
    Synchronous screen capture manager.
    """

    def __init__(
        self,
        monitor_index: int = 1,
        bbox: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize ScreenCapture.
        
        Args:
            monitor_index: Monitor index (1 = primary monitor in mss).
            bbox: Optional bounding box dict {'top': y, 'left': x, 'width': w, 'height': h}.
        """
        self.monitor_index = monitor_index
        self.bbox = bbox
        self._sct = mss.MSS() if HAS_MSS else None

        if self._sct:
            if monitor_index < len(self._sct.monitors):
                self.monitor = self._sct.monitors[monitor_index]
            else:
                self.monitor = self._sct.monitors[1]
        else:
            self.monitor = {"top": 0, "left": 0, "width": 1440, "height": 900}

        logger.info(f"ScreenCapture initialized on monitor {monitor_index} ({self.monitor['width']}x{self.monitor['height']})")

    def set_roi(self, x: int, y: int, width: int, height: int):
        """Sets the active Region of Interest (ROI) for subsequent grabs."""
        self.bbox = {
            "top": int(y),
            "left": int(x),
            "width": int(width),
            "height": int(height),
        }

    def clear_roi(self):
        """Clears ROI to capture full monitor."""
        self.bbox = None

    def grab(self) -> Optional[np.ndarray]:
        """
        Grabs screen pixels and returns a BGR NumPy array (H, W, 3).
        """
        if not HAS_MSS or self._sct is None:
            # Fallback blank frame
            w = self.bbox["width"] if self.bbox else self.monitor["width"]
            h = self.bbox["height"] if self.bbox else self.monitor["height"]
            return np.zeros((h, w, 3), dtype=np.uint8)

        target_area = self.bbox if self.bbox else self.monitor

        try:
            sct_img = self._sct.grab(target_area)
            # Fast zero-copy buffer to BGRA
            bgra = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((sct_img.height, sct_img.width, 4))
            # Convert BGRA to BGR
            bgr = bgra[:, :, :3]
            return bgr
        except Exception as e:
            logger.error(f"Screen grab failed: {e}")
            return None

    def close(self):
        """Closes the mss context."""
        if self._sct:
            self._sct.close()
            self._sct = None


class ThreadedScreenCapture:
    """
    Runs screen capture in a dedicated background thread, storing only the freshest
    frame in a single-slot buffer (0ms buffer lag for game loops).
    """

    def __init__(
        self,
        monitor_index: int = 1,
        bbox: Optional[Dict[str, int]] = None,
        target_fps: int = 30,
    ):
        self.sc = ScreenCapture(monitor_index=monitor_index, bbox=bbox)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the capture worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        # Warmup
        for _ in range(5):
            if self._latest_frame is not None:
                break
            time.sleep(0.02)
        logger.info("ThreadedScreenCapture worker started.")

    def _capture_worker(self):
        while self._running:
            t0 = time.perf_counter()
            frame = self.sc.grab()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame

            elapsed = time.perf_counter() - t0
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Optional[np.ndarray]:
        """Returns the freshest screen frame in 0.001ms."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def stop(self):
        """Stops the background worker thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.sc.close()
        logger.info("ThreadedScreenCapture worker stopped.")


class MockScreenCapture:
    """
    Injects pre-recorded or synthetic desktop frames for unit and E2E testing.
    """

    def __init__(self, initial_frame: Optional[np.ndarray] = None):
        self.current_frame = initial_frame

    def set_frame(self, frame: np.ndarray):
        self.current_frame = frame

    def grab(self) -> Optional[np.ndarray]:
        if self.current_frame is not None:
            return self.current_frame.copy()
        return np.zeros((800, 800, 3), dtype=np.uint8)

    def close(self):
        pass
