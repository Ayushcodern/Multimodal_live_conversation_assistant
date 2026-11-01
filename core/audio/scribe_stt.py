"""
ElevenLabs Scribe Speech-to-Text Integration Module.

Securely loads credentials from environment without logging or exposing keys.
Communicates with the official ElevenLabs Scribe STT endpoint to transcribe
genuine spoken microphone audio, with robust DNS resilience, connection retries,
and high-availability fallback via Groq Whisper.
"""

import io
import time
import socket
from typing import Optional, Tuple
import httpx
from elevenlabs.client import ElevenLabs
from config.settings import get_env_var, load_project_env

DEFAULT_SCRIBE_MODEL = "scribe_v1"
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Global patch to prevent Windows [Errno 11001] getaddrinfo failed on IPv6/dual-stack lookups
_orig_getaddrinfo = socket.getaddrinfo


def _resilient_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror as e:
        # On Windows, if dual-stack (family=0) or IPv6 fails with [Errno 11001], fallback to IPv4
        if family != socket.AF_INET:
            try:
                return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
            except Exception:
                pass
        raise e


# Apply resilient getaddrinfo once
if socket.getaddrinfo is not _resilient_getaddrinfo:
    socket.getaddrinfo = _resilient_getaddrinfo


class ScribeTranscriber:
    """Interacts with the real ElevenLabs Speech-to-Text API with fallback resilience."""

    def __init__(self, model_id: str = DEFAULT_SCRIBE_MODEL):
        self.model_id = model_id
        self._client: Optional[ElevenLabs] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the ElevenLabs client securely with resilient HTTP transport."""
        load_project_env()
        api_key = get_env_var("ELEVENLABS_API_KEY")
        if not api_key:
            self._client = None
            return

        try:
            custom_transport = httpx.HTTPTransport(retries=3)
            custom_http = httpx.Client(
                transport=custom_transport,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
            self._client = ElevenLabs(api_key=api_key, httpx_client=custom_http)
        except Exception:
            try:
                self._client = ElevenLabs(api_key=api_key)
            except Exception:
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Check whether the transcription service has either ElevenLabs or Groq configured."""
        load_project_env()
        has_eleven = bool(get_env_var("ELEVENLABS_API_KEY"))
        has_groq = bool(get_env_var("GROQ_API_KEY"))
        return has_eleven or has_groq

    def _transcribe_groq_fallback(
        self, audio_bytes: bytes, language_code: str = "eng"
    ) -> Tuple[bool, str, str]:
        """High-speed fallback transcription via Groq Whisper."""
        load_project_env()
        groq_api_key = get_env_var("GROQ_API_KEY")
        if not groq_api_key:
            return False, "", "Groq API key not configured for fallback transcription."

        try:
            headers = {"Authorization": f"Bearer {groq_api_key}"}
            files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
            # Map 3-letter language code to 2-letter for whisper if needed
            lang = "hi" if language_code in ("hin", "hi") else "en"
            data = {"model": GROQ_STT_MODEL, "language": lang}

            with httpx.Client(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
                res = client.post(GROQ_STT_URL, headers=headers, files=files, data=data)

            if res.status_code == 200:
                result_json = res.json()
                text = result_json.get("text", "").strip()
                return (
                    True,
                    text,
                    "Transcription completed via Groq Whisper fallback (ElevenLabs Scribe network unavailable).",
                )
            else:
                return False, "", f"Groq Whisper Fallback Error: HTTP {res.status_code} - {res.text[:120]}"

        except Exception as e:
            err_str = str(e)
            if "11001" in err_str or "getaddrinfo" in err_str:
                return (
                    False,
                    "",
                    "Network Connection Error: DNS resolution failed ([Errno 11001]). "
                    "Please check your internet/Wi-Fi connection.",
                )
            return False, "", f"Groq Fallback Error: {err_str}"

    def transcribe(
        self, audio_bytes: bytes, language_code: str = "eng"
    ) -> Tuple[bool, str, str]:
        """
        Send genuine captured audio bytes to ElevenLabs Scribe with automatic retry
        and seamless Groq Whisper fallback if DNS or connection errors occur.

        Args:
            audio_bytes: In-memory WAV audio bytes captured from the microphone.
            language_code: Target spoken language code (default 'eng' for English).

        Returns:
            Tuple[bool, str, str]:
                (Success status, transcription text or empty string, diagnostic/error message).
        """
        if not audio_bytes or len(audio_bytes) == 0:
            return False, "", "Transcription Error: Audio buffer is empty."

        if self._client is None:
            self._initialize_client()

        # If ElevenLabs API key is missing, attempt Groq directly
        if self._client is None:
            load_project_env()
            if get_env_var("GROQ_API_KEY"):
                return self._transcribe_groq_fallback(audio_bytes, language_code)
            return (
                False,
                "",
                "Configuration Error: ELEVENLABS_API_KEY is not set or invalid in environment. "
                "Please configure ELEVENLABS_API_KEY in your .env or api.env file.",
            )

        last_error = ""
        is_dns_error = False

        # Attempt ElevenLabs Scribe with up to 2 retries for transient DNS/connection glitches
        for attempt in range(2):
            try:
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "recording.wav"

                response = self._client.speech_to_text.convert(
                    file=audio_file,
                    model_id=self.model_id,
                    language_code=language_code,
                )

                transcript_text = getattr(response, "text", "")
                if not transcript_text and hasattr(response, "words"):
                    words_list = [w.text for w in getattr(response, "words", []) if hasattr(w, "text")]
                    transcript_text = " ".join(words_list)

                transcript_text = transcript_text.strip()
                return True, transcript_text, "Transcription completed successfully via ElevenLabs Scribe."

            except Exception as e:
                last_error = str(e)
                if "11001" in last_error or "getaddrinfo" in last_error:
                    is_dns_error = True
                    time.sleep(0.5)  # Short pause before retry
                    continue
                elif "connect" in last_error.lower() or "timeout" in last_error.lower():
                    time.sleep(0.5)
                    continue
                else:
                    break

        # If ElevenLabs Scribe failed (DNS resolution, network drop, or server issue),
        # automatically fall back to Groq Whisper
        load_project_env()
        if get_env_var("GROQ_API_KEY"):
            fb_success, fb_text, fb_msg = self._transcribe_groq_fallback(audio_bytes, language_code)
            if fb_success:
                return True, fb_text, fb_msg

        # If fallback also failed or unavailable, return a clean, actionable message
        if is_dns_error or "11001" in last_error or "getaddrinfo" in last_error:
            return (
                False,
                "",
                "Network Connection Error: DNS resolution failed ([Errno 11001]). "
                "Unable to reach transcription servers (api.elevenlabs.io). "
                "Please check your internet/Wi-Fi connection.",
            )

        return (
            False,
            "",
            f"ElevenLabs Scribe API Error: {last_error}",
        )
