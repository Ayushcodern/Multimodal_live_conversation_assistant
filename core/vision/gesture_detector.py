"""
Hand Landmark and Gesture Detector Module.

Uses Google MediaPipe HandLandmarker (Tasks Python API) to perform real-time
hand tracking on actual camera frames without simulated detections.
Calculates hand bounding box/centroid, extracts normalized geometric features,
and renders real landmark connections.
"""

import os
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

from core.vision.gesture_classifier import extract_hand_feature_vector

# Standard 21 hand landmark connections defined by MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base to pinky base
]

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "hand_landmarker.task"
)


class HandGestureDetector:
    """Detects visible hands, extracts actual 21 landmarks, and draws connections."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, num_hands: int = 2):
        self.model_path = model_path
        self.num_hands = num_hands
        self.detector = None
        self.is_initialized = False
        self.init_error: Optional[str] = None
        self._mp_image_module = None

        self._initialize_detector()

    def _initialize_detector(self):
        """Initializes the MediaPipe Tasks HandLandmarker."""
        if not os.path.exists(self.model_path):
            self.init_error = (
                f"Model file not found at: '{self.model_path}'.\n"
                "Please download the official Google MediaPipe 'hand_landmarker.task' asset "
                "into the 'models/' folder."
            )
            self.is_initialized = False
            return

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp

            self._mp_image_module = mp.Image
            self._mp_image_format = mp.ImageFormat

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.is_initialized = True
            self.init_error = None
        except Exception as e:
            self.detector = None
            self.is_initialized = False
            self.init_error = f"MediaPipe initialization failed: {str(e)}"

    def process_frame(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, str, List[Dict[str, Any]]]:
        """
        Process an RGB frame from the webcam.

        Returns:
            Tuple[annotated_rgb_frame, status_text, detected_hands_info]:
                - annotated_rgb_frame: Frame with landmarks and connections drawn.
                - status_text: e.g. "Hand detected (1)", "No hand detected", or error message.
                - detected_hands_info: List of dicts with real coordinates, centroid, and normalized feature vector.
        """
        if rgb_frame is None or rgb_frame.size == 0:
            return rgb_frame, "Camera unavailable", []

        if not self.is_initialized or self.detector is None:
            err_msg = self.init_error or "MediaPipe Hand Landmarker not initialized"
            annotated = rgb_frame.copy()
            cv2.putText(
                annotated,
                "Model asset missing: models/hand_landmarker.task",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 60, 60),
                2,
                cv2.LINE_AA,
            )
            return annotated, f"Initialization Error: {err_msg}", []

        h, w, _ = rgb_frame.shape

        try:
            mp_image = self._mp_image_module(
                image_format=self._mp_image_format.SRGB,
                data=rgb_frame
            )
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            return rgb_frame, f"Processing Error: {str(e)}", []

        annotated_frame = rgb_frame.copy()
        hand_landmarks_list = detection_result.hand_landmarks

        if not hand_landmarks_list:
            return annotated_frame, "No hand detected", []

        hands_info: List[Dict[str, Any]] = []

        for hand_idx, landmarks in enumerate(hand_landmarks_list):
            pixel_points = []
            xs = []
            ys = []

            for lm in landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pixel_points.append((px, py))
                xs.append(px)
                ys.append(py)

            # Draw connections (lines)
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(pixel_points) and end_idx < len(pixel_points):
                    cv2.line(
                        annotated_frame,
                        pixel_points[start_idx],
                        pixel_points[end_idx],
                        (0, 255, 128),
                        2,
                        cv2.LINE_AA,
                    )

            # Draw landmark keypoints (circles)
            for px, py in pixel_points:
                cv2.circle(annotated_frame, (px, py), 4, (255, 80, 0), -1, cv2.LINE_AA)
                cv2.circle(annotated_frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

            # Calculate actual position (centroid & bounding box)
            min_x, max_x = max(0, min(xs)), min(w - 1, max(xs))
            min_y, max_y = max(0, min(ys)), min(h - 1, max(ys))
            centroid_x = int(np.mean(xs))
            centroid_y = int(np.mean(ys))

            # Draw bounding box around hand
            cv2.rectangle(
                annotated_frame,
                (min_x - 5, min_y - 5),
                (max_x + 5, max_y + 5),
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

            # Draw centroid marker
            cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Label hand index and centroid
            cv2.putText(
                annotated_frame,
                f"Hand {hand_idx + 1} ({centroid_x}, {centroid_y})",
                (min_x, max(20, min_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Extract normalized feature vector from raw landmarks
            feat_vec = extract_hand_feature_vector(landmarks, image_width=w, image_height=h)

            hands_info.append({
                "hand_index": hand_idx + 1,
                "centroid": (centroid_x, centroid_y),
                "bbox": (min_x, min_y, max_x, max_y),
                "landmark_count": len(landmarks),
                "feature_vector": feat_vec,
            })

        count = len(hand_landmarks_list)
        status_text = f"Hand detected ({count})" if count > 0 else "No hand detected"
        return annotated_frame, status_text, hands_info
