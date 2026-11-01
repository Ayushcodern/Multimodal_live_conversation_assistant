"""
Camera Module for Real Webcam Capture.

This module provides thread-safe access to the physical webcam using OpenCV.
It strictly interfaces with real hardware devices and returns genuine frames,
surfacing errors if the device is disconnected or unavailable.
"""

import threading
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraManager:
    """Manages physical webcam lifecycle and real frame capture."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the camera manager.

        Args:
            camera_index: Hardware index of the webcam (default 0).
            width: Target frame width in pixels (downscaled to preserve i3 CPU/RAM).
            height: Target frame height in pixels.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """Return True if the camera stream is actively open."""
        with self._lock:
            return self._is_active and self._cap is not None and self._cap.isOpened()

    def start(self) -> Tuple[bool, str]:
        """
        Open the physical camera device.

        Returns:
            Tuple[bool, str]: (Success status, descriptive status/error message).
        """
        with self._lock:
            if self._is_active and self._cap is not None and self._cap.isOpened():
                return True, f"Camera (Index {self.camera_index}) is already connected."

            # cv2.CAP_DSHOW on Windows provides faster initialization and direct hardware access
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            # Fallback if DirectShow fails or index cannot open with backend
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                cap.release()
                self._cap = None
                self._is_active = False
                return (
                    False,
                    f"Camera Error: Could not open webcam at index {self.camera_index}. "
                    "Ensure a camera is connected, permissions are granted, and no other app is using it.",
                )

            # Configure resolution to be gentle on i3 hardware
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Test reading an initial real frame to verify hardware response
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                self._cap = None
                self._is_active = False
                return (
                    False,
                    f"Camera Error: Device at index {self.camera_index} opened but failed to deliver video frames.",
                )

            self._cap = cap
            self._is_active = True
            return True, f"Connected to camera (Index {self.camera_index})."

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Read the next raw frame from the real webcam.
        Converts BGR (OpenCV format) to RGB (standard format for web display).

        Returns:
            Tuple[bool, Optional[np.ndarray], str]:
                (success, rgb_frame or None, status/error message).
        """
        with self._lock:
            if not self._is_active or self._cap is None or not self._cap.isOpened():
                return False, None, "Camera is not active."

            ret, frame = self._cap.read()
            if not ret or frame is None:
                return (
                    False,
                    None,
                    "Camera Error: Failed to grab frame from device. The camera may have been disconnected.",
                )

            # Convert BGR to RGB for UI presentation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, rgb_frame, "OK"

    def release(self) -> Tuple[bool, str]:
        """
        Safely stop capturing and release camera hardware resources.

        Returns:
            Tuple[bool, str]: (Success status, confirmation message).
        """
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as exc:
                    self._cap = None
                    self._is_active = False
                    return False, f"Error releasing camera: {exc}"

            self._cap = None
            self._is_active = False
            return True, "Camera disconnected successfully."

