"""
ElevenLabs Scribe Speech-to-Text Integration Module.

Securely loads credentials from environment without logging or exposing keys.
Communicates with the official ElevenLabs Scribe STT endpoint to transcribe
genuine spoken microphone audio.
"""

import io
from typing import Optional, Tuple
from elevenlabs.client import ElevenLabs
from config.settings import get_env_var, load_project_env

DEFAULT_SCRIBE_MODEL = "scribe_v1"


class ScribeTranscriber:
    """Interacts with the real ElevenLabs Speech-to-Text API."""

    def __init__(self, model_id: str = DEFAULT_SCRIBE_MODEL):
        self.model_id = model_id
        self._client: Optional[ElevenLabs] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the ElevenLabs client securely."""
        load_project_env()
        api_key = get_env_var("ELEVENLABS_API_KEY")
        if not api_key:
            self._client = None
            return

        try:
            self._client = ElevenLabs(api_key=api_key)
        except Exception:
            self._client = None

    @property
    def is_configured(self) -> bool:
        """Check whether the ElevenLabs client is configured with an API key."""
        return self._client is not None

    def transcribe(
        self, audio_bytes: bytes, language_code: str = "eng"
    ) -> Tuple[bool, str, str]:
        """
        Send genuine captured audio bytes to ElevenLabs Scribe.

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

        if self._client is None:
            return (
                False,
                "",
                "Configuration Error: ELEVENLABS_API_KEY is not set or invalid in environment. "
                "Please configure ELEVENLABS_API_KEY in your .env or api.env file.",
            )

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
            error_details = str(e)
            return (
                False,
                "",
                f"ElevenLabs Scribe API Error: {error_details}",
            )
