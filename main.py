"""
Main Application Entrypoint for PS-16 Live Assistant.

Launches the Gradio interface and guarantees camera, microphone, and
reasoning resources are managed and released cleanly on shutdown.
"""

import atexit
import signal
import sys
from core.vision.camera import CameraManager
from core.vision.gesture_detector import HandGestureDetector
from core.vision.face_detector import FaceHeadDetector
from core.vision.gesture_classifier import PatientGestureClassifier
from core.audio.recorder import AudioRecorder
from core.audio.scribe_stt import ScribeTranscriber
from core.audio.flash_tts import FlashTTS
from core.llm.groq_assistant import GroqAssistant
from ui.app import create_app

# Hardware device singletons
camera_manager = CameraManager(camera_index=0, width=640, height=480)
gesture_detector = HandGestureDetector()
face_detector = FaceHeadDetector()
gesture_classifier = PatientGestureClassifier(similarity_threshold=0.88, temporal_window=5, min_stable_frames=3)
audio_recorder = AudioRecorder(sample_rate=16000, channels=1)
scribe_transcriber = ScribeTranscriber()
groq_assistant = GroqAssistant()
flash_tts = FlashTTS()


def cleanup():
    """Ensure hardware camera and audio streams are released when Python shuts down."""
    if camera_manager.is_active:
        print("\nReleasing webcam resources...")
        camera_manager.release()
    if audio_recorder.is_recording:
        print("Releasing microphone resources...")
        audio_recorder.release()
    print("Resources released safely.")


# Register cleanup on normal exit and interrupt signals
atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle termination signals cleanly."""
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Build and launch the Gradio user interface."""
    app = create_app(
        camera_manager=camera_manager,
        gesture_detector=gesture_detector,
        face_detector=face_detector,
        gesture_classifier=gesture_classifier,
        audio_recorder=audio_recorder,
        scribe_transcriber=scribe_transcriber,
        groq_assistant=groq_assistant,
        flash_tts=flash_tts,
    )
    print("Starting PS-16 Live Conversation Assistant (Phase 9: Patient Gesture Calibration)...")
    print(f"Active Reasoning Model: {groq_assistant.model_id}")
    print(f"Active TTS Model: {flash_tts.model_id}")
    print("Serving Gradio UI on http://127.0.0.1:7860")
    try:
        app.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except OSError:
        # Fallback to next available port if 7860 is temporarily busy
        print("Port 7860 in use; trying next available port in range 7861-7870...")
        app.launch(server_name="127.0.0.1", server_port=None, share=False)


if __name__ == "__main__":
    main()
