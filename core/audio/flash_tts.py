"""
ElevenLabs Flash Text-to-Speech Integration Module.

Securely loads credentials from environment without logging or exposing keys.
Communicates with the official ElevenLabs Flash TTS endpoint to synthesize speech
from explicitly confirmed patient messages.
"""

from typing import Optional, Tuple, Set
from elevenlabs.client import ElevenLabs
from config.settings import get_env_var, load_project_env

# Low-latency Flash model supporting English, Hindi, and 30+ languages
DEFAULT_FLASH_TTS_MODEL = "eleven_flash_v2_5"

# Verified eligible premade voice for ElevenLabs Free tier API usage:
# Roger (Laid-Back, Casual, Resonant) - confirmed via live API test
DEFAULT_FREE_PREMADE_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

# Standard premade voice IDs that are free-tier eligible on all ElevenLabs accounts
FREE_PREMADE_VOICE_IDS: Set[str] = {
    "CwhRBWXzGAHq8TQ4Fs17",  # Roger
    "EXAVITQu4vr4xnSDxMaL",  # Sarah
    "FGY2WhTYpPnrIDTdsKH5",  # Laura
    "IKne3meq5aSn9XLyUdCD",  # Charlie
    "JBFqnCBsd6RMkjVDRZzb",  # George
    "N2lVS1w4EtoT3dr4eOWO",  # Callum
    "SAz9YHcvj6GT2YYXdXww",  # River
    "SOYHLrjzK2X1ezoPC6cr",  # Harry
    "TX3LPaxmHKxFdv7VOQHJ",  # Liam
    "Xb7hH8MSUJpSbSDYk0k2",  # Alice
    "XrExE9yKIg1WjnnlVkGX",  # Matilda
    "bIHbv24MWmeRgasZH58o",  # Will
    "cgSgspJ2msm6clMCkdW9",  # Jessica
    "cjVigY5qzO86Huf0OWal",  # Eric
    "hpp4J3VqNfWAUOO0d1Us",  # Bella
    "iP95p4xoKVk53GoZ742B",  # Chris
    "nPczCjzI2devNBz1zQrb",  # Brian
    "onwK4e9ZLuTAKqWW03F9",  # Daniel
    "pFZP5JQG7iQjIQuC4Bku",  # Lily
    "pNInz6obpgDQGcFmaJgB",  # Adam
    "pqHfZKP75CvOlQylNhV4",  # Bill
}


class FlashTTS:
    """Interacts with the real ElevenLabs Text-to-Speech API."""

    def __init__(self, model_id: str = DEFAULT_FLASH_TTS_MODEL):
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
        """Check whether the client is configured with an API key."""
        return self._client is not None

    def get_effective_voice_id(self) -> Tuple[str, bool, Optional[str]]:
        """
        Determines the effective voice ID to use.
        Returns:
            Tuple[effective_voice_id, is_fallback, warning_message]
        """
        load_project_env()
        configured_id = get_env_var("ELEVENLABS_VOICE_ID")

        if not configured_id:
            return (
                DEFAULT_FREE_PREMADE_VOICE_ID,
                True,
                "Notice: ELEVENLABS_VOICE_ID is not configured. Defaulting to free-tier premade voice (Roger).",
            )

        clean_id = configured_id.strip()

        # If user configured an eligible premade voice, use it directly
        if clean_id in FREE_PREMADE_VOICE_IDS:
            return clean_id, False, None

        # The user configured a custom/library voice (such as Raju) which requires a paid Creator plan.
        # Fallback automatically to an eligible premade voice so synthesis succeeds on Free accounts
        return (
            DEFAULT_FREE_PREMADE_VOICE_ID,
            True,
            f"Notice: Configured voice '{clean_id}' is a restricted Library/Professional voice requiring a paid ElevenLabs plan.\n"
            f"Automatically routed to verified Free-tier premade voice 'Roger' ({DEFAULT_FREE_PREMADE_VOICE_ID}) to prevent 402 Payment Required errors.",
        )

    def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[bytes], str]:
        """
        Synthesize speech from confirmed patient text using ElevenLabs Flash TTS.

        Args:
            text: Confirmed text to speak aloud.
            voice_id: Optional voice ID override. Defaults to effective eligible voice ID.

        Returns:
            Tuple[bool, Optional[bytes], str]:
                (success status, MP3 audio bytes or None, status/error message)
        """
        if not text or not text.strip():
            return False, None, "TTS Error: Cannot synthesize empty text."

        clean_text = text.strip()

        if self._client is None:
            self._initialize_client()

        if self._client is None:
            return (
                False,
                None,
                "Configuration Error: ELEVENLABS_API_KEY is not configured in .env or api.env. "
                "Please configure a valid ElevenLabs API key.",
            )

        if voice_id:
            active_voice_id = voice_id
            routing_notice = None
        else:
            active_voice_id, is_fallback, routing_notice = self.get_effective_voice_id()

        try:
            # Call official ElevenLabs TTS convert endpoint
            audio_generator = self._client.text_to_speech.convert(
                voice_id=active_voice_id,
                text=clean_text,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )

            # Collect raw MP3 bytes in memory without persistent file creation
            audio_bytes = b"".join(audio_generator)
            if not audio_bytes or len(audio_bytes) == 0:
                return False, None, "TTS Error: ElevenLabs returned an empty audio stream."

            success_msg = f"Audio synthesized successfully ({len(audio_bytes)} bytes) using voice '{active_voice_id}'."
            if routing_notice:
                success_msg = f"{success_msg}\n{routing_notice}"

            return True, audio_bytes, success_msg

        except Exception as e:
            err_str = str(e)
            if "library_voice" in err_str or "paid_plan_required" in err_str or "402" in err_str:
                return (
                    False,
                    None,
                    f"ElevenLabs Error (402 Payment Required): Voice ID '{active_voice_id}' is a restricted Community/Library voice.\n"
                    "Free accounts can only use ElevenLabs premade voices via API (e.g. Roger: CwhRBWXzGAHq8TQ4Fs17).",
                )
            return False, None, f"ElevenLabs TTS Error: {err_str}"

    def check_cloning_entitlement(self) -> Tuple[bool, str]:
        """
        Queries ElevenLabs account entitlement for voice cloning.
        Returns (can_clone: bool, explanation: str).
        """
        if self._client is None:
            self._initialize_client()

        if self._client is None:
            return False, "ElevenLabs API key is not configured."

        try:
            user = self._client.user.get()
            sub = user.subscription
            can_clone = getattr(sub, "can_use_instant_voice_cloning", False)
            tier = getattr(sub, "tier", "unknown")
            if can_clone:
                return True, f"Account tier '{tier}' is entitled to instant voice cloning."
            else:
                return False, (
                    f"ElevenLabs account tier is '{tier}'. Instant Voice Cloning requires a paid subscription "
                    f"(Creator tier or above). Core application continues operating smoothly using verified premade voices."
                )
        except Exception as e:
            return False, f"Could not query ElevenLabs user account: {str(e)}"
