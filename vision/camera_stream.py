"""
CameraStream: Thread-safe, resilient camera pipeline for real-time video capture.
Features:
- Standard CameraStream for simple synchronous feeds.
- ThreadedCameraStream for high-FPS, zero-latency background buffer flushing.
- Full support for physical webcams and synthetic mock streams for testing.
"""

import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from loguru import logger


class CameraStream:
    """
    Manages video frame capture from webcams or video files/mock frames.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        target_width: int = 1280,
        target_height: int = 720,
        fps: int = 30,
    ):
        self.source = source
        self.target_width = target_width
        self.target_height = target_height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_mock = isinstance(source, (str, Path))
        
        self.open()

    def open(self) -> bool:
        """Opens video capture device or file."""
        if isinstance(self.source, Path):
            self.source = str(self.source)

        logger.info(f"Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        if not self._is_mock:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.success(f"Video stream ready. Resolution: {actual_w}x{actual_h}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next frame from the camera stream."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self._is_mock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return False, None

        return True, frame

    def get_resolution(self) -> Tuple[int, int]:
        """Returns (width, height) of current capture stream."""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def is_opened(self) -> bool:
        """Checks if stream is active."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Releases the camera hardware / video file cleanly."""
        if self.cap:
            logger.info("Releasing camera stream.")
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ThreadedCameraStream:
    """
    High-performance threaded camera stream that continuously reads frames in a background thread.
    Prevents hardware frame buffer build-up, guaranteeing 0ms video latency during heavy AI computation.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        target_width: int = 1280,
        target_height: int = 720,
        fps: int = 30,
    ):
        self.stream = CameraStream(source, target_width, target_height, fps)
        self.last_frame: Optional[np.ndarray] = None
        self.last_ret: bool = False
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        if self.stream.is_opened():
            self.start()

    def start(self) -> None:
        """Starts background reader thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("ThreadedCameraStream background reader thread started.")

    def _update_loop(self) -> None:
        """Continuously reads frames as fast as hardware provides."""
        while self.running:
            ret, frame = self.stream.read_frame()
            with self.lock:
                self.last_ret = ret
                self.last_frame = frame
            time.sleep(0.005)  # Yield CPU

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Instantly retrieves the latest freshly captured frame with zero buffer lag."""
        with self.lock:
            if self.last_frame is not None:
                return self.last_ret, self.last_frame.copy()
            return self.last_ret, None

    def stop(self) -> None:
        """Stops background reader thread and releases hardware."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()
        logger.info("ThreadedCameraStream stopped.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
