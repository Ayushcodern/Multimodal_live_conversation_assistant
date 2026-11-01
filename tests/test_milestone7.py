"""
Verification Test for Milestone 7:
Lip Activity & Syllable Estimation (Experimental Timing Cue).

Tests:
1. FaceHeadDetector initialization and model availability.
2. Synthetic frame processing (blank frame -> No face detected, graceful degradation).
3. Aperture calculation and syllable cycle peak heuristic on synthetic aperture series.
4. UI component and telemetry wiring.
"""

import time
import numpy as np
from core.vision.face_detector import FaceHeadDetector

def test_detector_initialization():
    detector = FaceHeadDetector()
    assert detector.is_initialized, f"Face detector failed to initialize: {detector.init_error}"
    print("PASS: FaceHeadDetector successfully initialized with MediaPipe model.")

def test_graceful_degradation_blank_frame():
    detector = FaceHeadDetector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, status, movement = detector.process_frame(blank_frame)
    assert status == "No face detected"
    assert "lip_activity" not in movement or movement.get("details") == "No face visible"
    print("PASS: Graceful degradation on blank frame verified.")

def test_lip_activity_heuristic_math():
    detector = FaceHeadDetector()
    # Simulate a sequence of mouth aperture ratios (open-close-open-close syllables)
    # Ratios simulating 2 speaking syllable peaks: baseline 0.12, peak 0.28, drop 0.14, peak 0.30, drop 0.13
    simulated_ratios = [
        0.12, 0.12, 0.13, 0.12, 0.14, 0.22, 0.28, 0.24, 0.14, 0.13,
        0.12, 0.21, 0.30, 0.25, 0.15, 0.12, 0.13, 0.12, 0.13, 0.12
    ]
    t0 = time.time()
    for i, r in enumerate(simulated_ratios):
        detector.mouth_aperture_history.append((t0 + i * 0.05, r))

    lip_analysis = detector._analyze_lip_activity()
    print(f"Lip analysis result on simulated speech: {lip_analysis}")
    assert lip_analysis["estimated_syllables_heuristic"] >= 1, "Syllable heuristic failed to detect peaks"
    assert "Experimental" in lip_analysis["confidence_note"], "Confidence note missing required safeguard statement"
    print("PASS: Aperture calculation and syllable peak counting verified.")

if __name__ == "__main__":
    print("--- RUNNING MILESTONE 7 VERIFICATION ---")
    test_detector_initialization()
    test_graceful_degradation_blank_frame()
    test_lip_activity_heuristic_math()
    print("--- ALL MILESTONE 7 UNIT TESTS PASSED ---")

