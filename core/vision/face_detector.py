"""
Facial Landmark and Head Movement Detector Module.

Uses Google MediaPipe FaceLandmarker (Tasks Python API) to perform real-time
facial tracking on actual camera frames without simulated detections.
Calculates real temporal head movements (Left/Right, Up/Down, possible Nod, possible Shake)
using nose tip and facial landmark displacement over a rolling time window.
"""

import os
import time
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

DEFAULT_FACE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "face_landmarker.task",
)

# Key landmark indices in MediaPipe Face Mesh (478 landmarks):
# Nose tip: 1
# Chin: 152
# Left cheek edge: 234
# Right cheek edge: 454
# Forehead: 10
# Left eye outer: 33
# Right eye outer: 263
NOSE_TIP_IDX = 1
CHIN_IDX = 152
FOREHEAD_IDX = 10
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454

# Lip landmark indices for mouth aperture & syllable estimation (Milestone 7)
# Upper lip inner: 13, Lower lip inner: 14
# Upper lip outer: 0, Lower lip outer: 17
# Left corner of mouth: 61, Right corner of mouth: 291
UPPER_LIP_INNER_IDX = 13
LOWER_LIP_INNER_IDX = 14
UPPER_LIP_OUTER_IDX = 0
LOWER_LIP_OUTER_IDX = 17
MOUTH_CORNER_LEFT_IDX = 61
MOUTH_CORNER_RIGHT_IDX = 291

# Subset of facial contour points for lightweight rendering on Intel i3
FACIAL_CONTOUR_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]


class FaceHeadDetector:
    """Detects visible face landmarks and calculates temporal head movement signals."""

    def __init__(
        self,
        model_path: str = DEFAULT_FACE_MODEL_PATH,
        history_len: int = 15,
        movement_threshold_px: float = 8.0,
    ):
        """
        Initialize the Face and Head Movement detector.

        Args:
            model_path: Path to the official face_landmarker.task file.
            history_len: Number of recent frames stored for temporal movement analysis.
            movement_threshold_px: Pixel displacement threshold to detect head movement.
        """
        self.model_path = model_path
        self.history_len = history_len
        self.movement_threshold_px = movement_threshold_px

        self.detector = None
        self.is_initialized = False
        self.init_error: Optional[str] = None
        self._mp_image_module = None
        self._mp_image_format = None

        # Rolling history of (timestamp, nose_x, nose_y)
        self.trajectory: deque = deque(maxlen=history_len)

        # Rolling history of mouth aperture ratios for syllable estimation (Milestone 7)
        self.mouth_aperture_history: deque = deque(maxlen=30)
        self.last_lip_activity_timestamp: float = 0.0

        self._initialize_detector()

    def _initialize_detector(self):
        """Initializes the MediaPipe Tasks FaceLandmarker."""
        if not os.path.exists(self.model_path):
            self.init_error = (
                f"Face model file not found at: '{self.model_path}'.\n"
                "Please download the official Google MediaPipe 'face_landmarker.task' asset "
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
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.is_initialized = True
            self.init_error = None
        except Exception as e:
            self.detector = None
            self.is_initialized = False
            self.init_error = f"MediaPipe Face initialization failed: {str(e)}"

    def _analyze_temporal_movement(self) -> Dict[str, Any]:
        """
        Analyze nose displacement trajectory over the rolling time window.
        Returns experimental directional displacement and potential oscillation patterns.
        """
        if len(self.trajectory) < 6:
            return {
                "direction_h": "Center",
                "direction_v": "Center",
                "delta_x": 0.0,
                "delta_y": 0.0,
                "potential_nod": False,
                "potential_shake": False,
                "details": "Buffering motion trajectory...",
            }

        xs = [pt[1] for pt in self.trajectory]
        ys = [pt[2] for pt in self.trajectory]

        # Recent displacement: compare mean of older half with mean of newer half
        half = len(self.trajectory) // 2
        older_xs = xs[:half]
        newer_xs = xs[half:]
        older_ys = ys[:half]
        newer_ys = ys[half:]

        delta_x = np.mean(newer_xs) - np.mean(older_xs)
        delta_y = np.mean(newer_ys) - np.mean(older_ys)

        # Horizontal direction
        if delta_x > self.movement_threshold_px:
            dir_h = "Moving Right"
        elif delta_x < -self.movement_threshold_px:
            dir_h = "Moving Left"
        else:
            dir_h = "Center"

        # Vertical direction (Note: in image coordinates, y increases downward)
        if delta_y > self.movement_threshold_px:
            dir_v = "Moving Down"
        elif delta_y < -self.movement_threshold_px:
            dir_v = "Moving Up"
        else:
            dir_v = "Center"

        # Check for oscillation / sign changes in velocity for potential nod/shake
        diffs_x = np.diff(xs)
        diffs_y = np.diff(ys)

        # Count zero-crossings (reversals) with significant amplitude
        reversals_x = 0
        for i in range(len(diffs_x) - 1):
            if (diffs_x[i] * diffs_x[i + 1] < 0) and (abs(diffs_x[i]) > 1.5 or abs(diffs_x[i + 1]) > 1.5):
                reversals_x += 1

        reversals_y = 0
        for i in range(len(diffs_y) - 1):
            if (diffs_y[i] * diffs_y[i + 1] < 0) and (abs(diffs_y[i]) > 1.5 or abs(diffs_y[i + 1]) > 1.5):
                reversals_y += 1

        # Heuristic for repeated oscillation: at least 2 reversals in short window
        potential_shake = reversals_x >= 2 and (max(xs) - min(xs)) > (self.movement_threshold_px * 1.5)
        potential_nod = reversals_y >= 2 and (max(ys) - min(ys)) > (self.movement_threshold_px * 1.5)

        summary_parts = []
        if dir_h != "Center":
            summary_parts.append(dir_h)
        if dir_v != "Center":
            summary_parts.append(dir_v)
        if potential_nod:
            summary_parts.append("Possible Nod detected (Experimental)")
        if potential_shake:
            summary_parts.append("Possible Shake detected (Experimental)")

        details = "; ".join(summary_parts) if summary_parts else "Steady"

        return {
            "direction_h": dir_h,
            "direction_v": dir_v,
            "delta_x": round(float(delta_x), 1),
            "delta_y": round(float(delta_y), 1),
            "potential_nod": bool(potential_nod),
            "potential_shake": bool(potential_shake),
            "details": details,
        }

    def _analyze_lip_activity(self) -> Dict[str, Any]:
        """
        Calculates mouth opening/activity over a rolling window.
        Provides an experimental syllable count heuristic based on aperture peak cycles.
        IMPORTANT: This is purely a timing and syllable-length cue, NOT lip-reading.
        """
        if len(self.mouth_aperture_history) < 10:
            return {
                "lip_state": "Buffering...",
                "current_aperture_ratio": 0.0,
                "is_speaking_activity": False,
                "estimated_syllables_heuristic": 0,
                "confidence_note": "Aperture window buffering (<10 frames)",
            }

        apertures = [pt[1] for pt in self.mouth_aperture_history]
        curr_ratio = apertures[-1]
        mean_aperture = float(np.mean(apertures))
        std_aperture = float(np.std(apertures))

        # Dynamic speaking activity threshold: significant aperture above baseline
        is_active = (curr_ratio > 0.18) or (std_aperture > 0.04 and curr_ratio > mean_aperture)

        # Estimate syllable cycles via peak counting in aperture history
        # A syllable corresponds to an open-close articulation cycle
        peaks = 0
        threshold = max(0.16, mean_aperture + 0.5 * std_aperture)
        for i in range(1, len(apertures) - 1):
            if apertures[i] > threshold and apertures[i] > apertures[i - 1] and apertures[i] > apertures[i + 1]:
                peaks += 1

        lip_state = "Speaking / Moving" if is_active else "Resting / Closed"

        return {
            "lip_state": lip_state,
            "current_aperture_ratio": round(float(curr_ratio), 3),
            "is_speaking_activity": bool(is_active),
            "estimated_syllables_heuristic": int(peaks),
            "confidence_note": "Experimental timing heuristic only. NOT decoded speech or lip-reading.",
        }

    def process_frame(
        self, rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Process an RGB frame to detect face landmarks and head movement.

        Args:
            rgb_frame: NumPy array (H, W, 3) in RGB format.

        Returns:
            Tuple[annotated_rgb_frame, face_status, movement_data]:
                - annotated_rgb_frame: Frame with face landmarks overlaid.
                - face_status: "Face detected", "No face detected", or initialization error.
                - movement_data: Real temporal motion signals.
        """
        if rgb_frame is None or rgb_frame.size == 0:
            self.trajectory.clear()
            return rgb_frame, "Camera unavailable", {"details": "No frame received"}

        if not self.is_initialized or self.detector is None:
            self.trajectory.clear()
            err_msg = self.init_error or "MediaPipe FaceLandmarker not initialized"
            return rgb_frame, f"Initialization Error: {err_msg}", {"details": "Model uninitialized"}

        h, w, _ = rgb_frame.shape

        try:
            mp_image = self._mp_image_module(
                image_format=self._mp_image_format.SRGB,
                data=rgb_frame,
            )
            result = self.detector.detect(mp_image)
        except Exception as e:
            self.trajectory.clear()
            return rgb_frame, f"Processing Error: {str(e)}", {"details": "Detection exception"}

        annotated_frame = rgb_frame.copy()
        face_landmarks_list = result.face_landmarks

        if not face_landmarks_list:
            self.trajectory.clear()
            return annotated_frame, "No face detected", {"details": "No face visible"}

        # Process primary face
        landmarks = face_landmarks_list[0]
        total_lms = len(landmarks)

        # Extract nose tip for head trajectory
        if NOSE_TIP_IDX < total_lms:
            nose_lm = landmarks[NOSE_TIP_IDX]
            nx = int(nose_lm.x * w)
            ny = int(nose_lm.y * h)
            self.trajectory.append((time.time(), nx, ny))

            # Draw nose tip reference tracker
            cv2.circle(annotated_frame, (nx, ny), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # Draw facial contour points
        for idx in FACIAL_CONTOUR_INDICES:
            if idx < total_lms:
                lm = landmarks[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(annotated_frame, (px, py), 2, (180, 220, 0), -1, cv2.LINE_AA)

        # Draw key alignment landmarks (Forehead, Chin, Cheeks)
        key_indices = [FOREHEAD_IDX, CHIN_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX]
        for idx in key_indices:
            if idx < total_lms:
                lm = landmarks[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(annotated_frame, (px, py), 3, (255, 100, 255), -1, cv2.LINE_AA)

        # Extract Lip Aperture Ratio (Milestone 7)
        if (UPPER_LIP_INNER_IDX < total_lms and LOWER_LIP_INNER_IDX < total_lms and
            MOUTH_CORNER_LEFT_IDX < total_lms and MOUTH_CORNER_RIGHT_IDX < total_lms):
            u_lip = landmarks[UPPER_LIP_INNER_IDX]
            l_lip = landmarks[LOWER_LIP_INNER_IDX]
            c_left = landmarks[MOUTH_CORNER_LEFT_IDX]
            c_right = landmarks[MOUTH_CORNER_RIGHT_IDX]

            # Vertical lip distance vs horizontal mouth width (scale-invariant ratio)
            vert_dist = np.linalg.norm(np.array([u_lip.x * w, u_lip.y * h]) - np.array([l_lip.x * w, l_lip.y * h]))
            horiz_dist = np.linalg.norm(np.array([c_left.x * w, c_left.y * h]) - np.array([c_right.x * w, c_right.y * h]))
            aperture_ratio = float(vert_dist / (horiz_dist + 1e-4))
            self.mouth_aperture_history.append((time.time(), aperture_ratio))
            self.last_lip_activity_timestamp = time.time()

            # Draw visual lip markers
            cv2.circle(annotated_frame, (int(u_lip.x * w), int(u_lip.y * h)), 2, (0, 165, 255), -1)
            cv2.circle(annotated_frame, (int(l_lip.x * w), int(l_lip.y * h)), 2, (0, 165, 255), -1)

        movement_data = self._analyze_temporal_movement()
        lip_data = self._analyze_lip_activity()
        movement_data["lip_activity"] = lip_data
        face_status = "Face detected"

        return annotated_frame, face_status, movement_data

