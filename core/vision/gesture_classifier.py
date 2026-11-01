"""
Patient-Specific Gesture Feature Extraction and Classifier Module.

Extracts scale- and translation-invariant geometric features from genuine
MediaPipe 21 hand landmarks:
- Finger extension & flexion states (thumb, index, middle, ring, pinky)
- Normalized pairwise fingertip-to-wrist and fingertip-to-palm distances
- Palm orientation angle
- Cosine similarity matching against patient-calibrated samples
- Multi-frame temporal smoothing to avoid single-frame flickering
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import numpy as np


def extract_hand_feature_vector(landmarks: List[Any], image_width: int = 640, image_height: int = 480) -> Optional[np.ndarray]:
    """
    Extracts a scale- and translation-invariant normalized feature vector (24 dimensions)
    from 21 MediaPipe hand landmarks.

    Landmarks format: list of objects with x, y, z or tuples (x, y, z).
    """
    if not landmarks or len(landmarks) < 21:
        return None

    # Convert landmarks to pixel / metric coordinate array (21, 3)
    pts = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        if hasattr(lm, "x"):
            pts[i] = [lm.x * image_width, lm.y * image_height, getattr(lm, "z", 0.0) * image_width]
        elif isinstance(lm, (list, tuple)):
            pts[i] = [lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0]

    # Wrist is origin (index 0)
    wrist = pts[0]
    centered = pts - wrist

    # Normalize by palm size (distance between wrist 0 and middle finger MCP 9)
    palm_scale = np.linalg.norm(centered[9])
    if palm_scale < 1e-4:
        palm_scale = 1.0
    norm_pts = centered / palm_scale

    features = []

    # 1. Normalized distances from wrist (0) to 5 fingertips (4, 8, 12, 16, 20)
    fingertip_indices = [4, 8, 12, 16, 20]
    for tip_idx in fingertip_indices:
        d = np.linalg.norm(norm_pts[tip_idx])
        features.append(d)

    # 2. Normalized pairwise distances between adjacent fingertips
    # (Thumb-Index: 4-8, Index-Middle: 8-12, Middle-Ring: 12-16, Ring-Pinky: 16-20, Thumb-Pinky: 4-20)
    tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]
    for i1, i2 in tip_pairs:
        d = np.linalg.norm(norm_pts[i1] - norm_pts[i2])
        features.append(d)

    # 3. Finger extension ratios (distance from MCP to tip vs MCP to PIP)
    # Fingers: Thumb (2-4 vs 1-2), Index (5-8 vs 5-6), Middle (9-12 vs 9-10), Ring (13-16 vs 13-14), Pinky (17-20 vs 17-18)
    finger_joints = [
        (2, 4, 1, 2),    # Thumb
        (5, 8, 5, 6),    # Index
        (9, 12, 9, 10),  # Middle
        (13, 16, 13, 14),# Ring
        (17, 20, 17, 18),# Pinky
    ]
    for mcp, tip, p1, p2 in finger_joints:
        len_total = np.linalg.norm(norm_pts[tip] - norm_pts[mcp])
        len_base = np.linalg.norm(norm_pts[p2] - norm_pts[p1])
        ratio = len_total / (len_base + 1e-4)
        features.append(ratio)

    # 4. Relative Y displacement (extension upwards vs curled downwards relative to MCP)
    for tip_idx, mcp_idx in [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]:
        dy = (norm_pts[tip_idx][1] - norm_pts[mcp_idx][1])  # Negative if tip is higher than MCP in image
        features.append(dy)

    # 5. Hand tilt angle in 2D plane (wrist to middle MCP vector)
    v_palm = norm_pts[9][:2]
    angle = np.arctan2(v_palm[1], v_palm[0])
    features.append(angle / np.pi)  # Normalized to [-1, 1]

    # Combine into feature vector
    feat_vec = np.array(features, dtype=np.float32)

    # L2 normalize feature vector for cosine distance computation
    norm = np.linalg.norm(feat_vec)
    if norm > 1e-4:
        feat_vec = feat_vec / norm

    return feat_vec


class CalibratedGesture:
    """Stores calibration samples for a single patient-defined gesture."""

    def __init__(self, name: str, meaning: str):
        self.name = name.strip()
        self.meaning = meaning.strip()
        self.samples: List[np.ndarray] = []

    def add_sample(self, feature_vector: np.ndarray):
        """Append a real landmark feature vector sample."""
        self.samples.append(feature_vector.copy())

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def mean_feature_vector(self) -> Optional[np.ndarray]:
        """Calculates the centroid feature vector across all captured calibration samples."""
        if not self.samples:
            return None
        centroid = np.mean(self.samples, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 1e-4 else centroid

    def consistency_score(self) -> float:
        """
        Calculates internal consistency (0.0 to 1.0) among calibration samples.
        Higher consistency means the patient held the gesture stably during calibration.
        """
        if len(self.samples) < 2:
            return 1.0 if len(self.samples) == 1 else 0.0

        centroid = self.mean_feature_vector()
        similarities = [np.dot(s, centroid) for s in self.samples]
        return float(np.clip(np.mean(similarities), 0.0, 1.0))


class PatientGestureClassifier:
    """
    Lightweight feature-based gesture classifier for Intel i3 CPU.
    Compares live extracted features with calibrated gesture templates
    using cosine similarity with temporal stability smoothing.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.88,
        temporal_window: int = 5,
        min_stable_frames: int = 3,
    ):
        """
        Args:
            similarity_threshold: Minimum cosine similarity required to match a gesture (0.0-1.0).
            temporal_window: Number of recent frames checked for stable recognition.
            min_stable_frames: Minimum required matching frames in the temporal window.
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window
        self.min_stable_frames = min_stable_frames

        # Dictionary of patient-calibrated gestures: {gesture_name: CalibratedGesture}
        self.gestures: Dict[str, CalibratedGesture] = {}

        # Recent frame predictions for temporal smoothing: deque of gesture_names or "Unknown"
        self._prediction_history: deque = deque(maxlen=temporal_window)

    def add_gesture(self, name: str, meaning: str) -> CalibratedGesture:
        """Register a new patient-specific gesture container."""
        g = CalibratedGesture(name=name, meaning=meaning)
        self.gestures[name] = g
        return g

    def remove_gesture(self, name: str) -> bool:
        """Delete a calibrated gesture."""
        if name in self.gestures:
            del self.gestures[name]
            self._prediction_history.clear()
            return True
        return False

    def clear_all(self):
        """Remove all patient calibrations."""
        self.gestures.clear()
        self._prediction_history.clear()

    def list_gestures(self) -> List[Dict[str, Any]]:
        """Return metadata of all calibrated gestures."""
        res = []
        for name, g in self.gestures.items():
            res.append({
                "name": g.name,
                "meaning": g.meaning,
                "sample_count": g.sample_count,
                "consistency": round(g.consistency_score() * 100, 1),
            })
        return res

    def classify_frame(
        self, feature_vector: Optional[np.ndarray]
    ) -> Tuple[str, Optional[str], float, str]:
        """
        Classifies a single hand feature vector using cosine similarity and temporal stability.

        Returns:
            Tuple[recognition_status, recognized_meaning, confidence_score, status_message]:
                - recognition_status: e.g. "Recognized: <Name>", "Uncertain", "No hand detected"
                - recognized_meaning: Meaning string assigned to the gesture, or None
                - confidence_score: Raw similarity metric (0.0 to 1.0)
                - status_message: Diagnostic note
        """
        if feature_vector is None:
            self._prediction_history.append("No Hand")
            return "No hand detected", None, 0.0, "Hand is not in view."

        if not self.gestures:
            self._prediction_history.clear()
            return "No calibrated gestures", None, 0.0, "No patient gestures calibrated yet."

        # Find closest calibrated gesture centroid
        best_name = None
        best_similarity = -1.0

        for name, gesture in self.gestures.items():
            if gesture.sample_count == 0:
                continue
            centroid = gesture.mean_feature_vector()
            if centroid is not None:
                sim = float(np.dot(feature_vector, centroid))
                if sim > best_similarity:
                    best_similarity = sim
                    best_name = name

        # Check raw similarity threshold
        instant_match = None
        if best_similarity >= self.similarity_threshold and best_name is not None:
            instant_match = best_name
        else:
            instant_match = "Uncertain"

        self._prediction_history.append(instant_match)

        # Temporal stability check: must appear at least min_stable_frames in recent history
        count_in_history = self._prediction_history.count(best_name)
        is_temporally_stable = count_in_history >= self.min_stable_frames

        if is_temporally_stable and instant_match == best_name:
            meaning = self.gestures[best_name].meaning
            msg = f"Consistent match for '{best_name}' ({best_similarity:.2f} similarity across {count_in_history}/{len(self._prediction_history)} frames)."
            return f"Recognized: {best_name}", meaning, round(best_similarity, 3), msg
        elif best_similarity >= self.similarity_threshold:
            return "Uncertain (Stabilizing)", None, round(best_similarity, 3), f"Detected '{best_name}' but stabilizing ({count_in_history}/{self.min_stable_frames} frames)..."
        else:
            return "Uncertain", None, round(max(0.0, best_similarity), 3), f"No calibrated gesture matched (best similarity: {best_similarity:.2f} < threshold {self.similarity_threshold})."

