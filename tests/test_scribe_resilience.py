"""
Unit tests for ScribeTranscriber DNS resilience and fallback handling.
Verifies that:
1. Normal transcription operates as expected.
2. When ElevenLabs Scribe encounters [Errno 11001] getaddrinfo failed,
   the transcriber automatically falls back to Groq Whisper and succeeds.
3. If both services fail, a clean, user-friendly diagnostic error is returned.
"""

import io
import wave
from unittest.mock import MagicMock
from core.audio.scribe_stt import ScribeTranscriber


def make_dummy_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00' * 16000)
    return buf.getvalue()


def test_normal_transcription():
    stt = ScribeTranscriber()
    wav = make_dummy_wav()
    success, text, msg = stt.transcribe(wav)
    print("Normal transcription result:", success, repr(text), msg)
    assert success is True
    print("PASS: Normal transcription test passed.")


def test_fallback_on_dns_failure():
    stt = ScribeTranscriber()
    wav = make_dummy_wav()

    # Simulate ElevenLabs throwing [Errno 11001] getaddrinfo failed
    stt._client.speech_to_text.convert = MagicMock(
        side_effect=Exception("[Errno 11001] getaddrinfo failed")
    )

    success, text, msg = stt.transcribe(wav)
    print("Simulated DNS failure result:", success, repr(text), msg)
    assert success is True, f"Expected fallback to succeed, got msg: {msg}"
    assert "Groq Whisper fallback" in msg, f"Expected Groq fallback notice in msg: {msg}"
    print("PASS: Automatic Groq Whisper fallback on [Errno 11001] passed.")


def test_clean_error_on_total_network_failure():
    stt = ScribeTranscriber()
    wav = make_dummy_wav()

    # Simulate ElevenLabs failure
    stt._client.speech_to_text.convert = MagicMock(
        side_effect=Exception("[Errno 11001] getaddrinfo failed")
    )
    # Simulate Groq fallback also failing
    stt._transcribe_groq_fallback = MagicMock(
        return_value=(False, "", "Network Connection Error: DNS resolution failed ([Errno 11001]). Please check your internet/Wi-Fi connection.")
    )

    success, text, msg = stt.transcribe(wav)
    assert success is False
    assert "[Errno 11001]" in msg
    assert "Please check your internet" in msg
    print("PASS: Clean error reporting on total network failure passed.")


if __name__ == "__main__":
    print("--- RUNNING SCRIBE RESILIENCE TESTS ---")
    test_normal_transcription()
    test_fallback_on_dns_failure()
    test_clean_error_on_total_network_failure()
    print("--- ALL SCRIBE RESILIENCE TESTS PASSED ---")
