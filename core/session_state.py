"""
Session State Model for PS-16 Multimodal Live Conversation Assistant.
Phase 10: Structured Input Context, Input Freshness Tracking, and Refined Response Metadata.

Tracks:
- Explicit timestamp and freshness of every modality (Speech, Hands, Face, Calibrated Gestures).
- Distinct modality availability indicators.
- Refined Groq interpretation metadata (reasoning summary, used modalities, uncertainty notes).
- Strict patient confirmation lifecycle with complete invalidation upon any edit/retry.
- Chronological conversation log with modalities used and spoken status.
"""

import time
import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ModalityFreshness(BaseModel):
    """Tracks status and timestamp of a sensor modality."""
    is_available: bool = False
    is_stale: bool = True
    last_updated_time: float = 0.0
    status_summary: str = "Unavailable"

    def update(self, available: bool, summary: str):
        self.is_available = available
        self.is_stale = not available
        self.last_updated_time = time.time()
        self.status_summary = summary

    def check_staleness(self, max_age_seconds: float = 4.0) -> bool:
        """Returns True if the sensor data was updated too long ago."""
        if not self.is_available:
            self.is_stale = True
            return True
        age = time.time() - self.last_updated_time
        self.is_stale = age > max_age_seconds
        return self.is_stale


class ConversationLogItem(BaseModel):
    """Represents a single confirmed dialogue turn in the conversation log."""
    speaker: str
    original_transcript: str = ""
    interpreted_message: str = ""
    confirmed_message: str
    modalities_used: List[str] = Field(default_factory=list)
    was_spoken: bool = False
    timestamp: str = ""


class QuickPhraseGroup(BaseModel):
    """Predefined and custom phrase groups for instant communication without STT/LLM."""
    basic: List[str] = Field(
        default_factory=lambda: [
            "Yes.",
            "No.",
            "Please wait.",
            "Please repeat.",
            "Thank you.",
        ]
    )
    needs: List[str] = Field(
        default_factory=lambda: [
            "I need water.",
            "I am hungry.",
            "I need my medicine.",
            "I need to use the washroom.",
            "I am tired.",
            "I feel hot.",
            "I feel cold.",
        ]
    )
    help: List[str] = Field(
        default_factory=lambda: [
            "I need help.",
            "I am in pain.",
            "Please call someone.",
            "Please stop.",
        ]
    )
    custom: List[str] = Field(default_factory=list)


class PatientProfile(BaseModel):
    """Patient-controlled communication profile."""
    enabled: bool = True
    display_name: str = ""
    preferred_language: str = "Auto"  # Auto, English, Hindi, Hinglish
    preferred_script: str = "Roman"   # Roman, Devanagari
    common_needs: List[str] = Field(
        default_factory=lambda: ["Water", "Food", "Medicine", "Rest", "Washroom"]
    )
    frequently_used_phrases: List[str] = Field(
        default_factory=lambda: [
            "Please give me water",
            "I need my medicine",
            "Please call someone",
            "I am feeling tired",
            "Thank you",
        ]
    )
    communication_notes: str = ""


class VoiceConsentRecord(BaseModel):
    """Enforceable consent record for personalized voice creation."""
    has_consented: bool = False
    signer_name: str = ""
    acknowledgement_text: str = ""
    timestamp: str = ""
    provider_voice_id: Optional[str] = None
    voice_type: str = "premade"  # premade, designed, cloned
    voice_label: str = "Roger (Premade Free-Tier Voice)"


class SessionState(BaseModel):
    """
    Per-client isolated session state stored inside Gradio gr.State.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Voice Consent & Personalization (Milestone 5)
    voice_consent: VoiceConsentRecord = Field(default_factory=VoiceConsentRecord)

    # Adaptive Patient Profile (Milestone 3)
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)

    # Quick Communication Panel (Milestone 4)
    quick_phrases: QuickPhraseGroup = Field(default_factory=QuickPhraseGroup)

    # Audio capture & transcription state
    current_speaker: str = "Patient"
    latest_patient_transcript: str = ""
    latest_partner_transcript: str = ""
    latest_raw_transcript: str = ""
    audio_recorded_bytes: Optional[bytes] = None
    speech_freshness: ModalityFreshness = Field(default_factory=ModalityFreshness)

    # Real vision sensor features (ephemeral & compact)
    latest_hand_features: List[Dict[str, Any]] = Field(default_factory=list)
    latest_head_features: Dict[str, Any] = Field(default_factory=dict)
    face_status: str = "Camera unavailable"
    hand_status: str = "Camera unavailable"
    is_vision_stale: bool = True
    vision_freshness: ModalityFreshness = Field(default_factory=ModalityFreshness)

    # Patient-specific gesture calibration & live recognition
    calibrated_gestures: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    calibration_samples_store: Dict[str, List[Any]] = Field(default_factory=dict)
    active_calibration_gesture_name: Optional[str] = None
    latest_recognized_gesture: str = "No calibrated gestures"
    latest_recognized_meaning: Optional[str] = None
    latest_gesture_confidence: float = 0.0
    gesture_freshness: ModalityFreshness = Field(default_factory=ModalityFreshness)

    # Refined AI interpretation & candidates from Groq
    latest_interpreted_message: str = ""
    ranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    selected_candidate_rank: Optional[int] = None
    alternative_interpretations: List[str] = Field(default_factory=list)
    reply_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    selected_reply_category: Optional[str] = None
    detected_language: str = "English"
    reasoning_summary: str = ""
    used_modalities: List[str] = Field(default_factory=list)
    uncertainty_notes: Optional[str] = None
    requires_clarification: bool = False
    clarification_question: Optional[str] = None

    # Strict patient confirmation lifecycle
    proposed_message: str = ""
    is_confirmed: bool = False
    confirmed_message: str = ""

    # TTS audio state
    latest_tts_audio_path: Optional[str] = None
    tts_status: str = "Awaiting confirmed message."

    # Chronological session dialogue log
    conversation_log: List[ConversationLogItem] = Field(default_factory=list)

    def invalidate_confirmation(self):
        """Invalidates confirmation whenever text is edited, rejected, or re-interpreted."""
        self.is_confirmed = False
        self.confirmed_message = ""
        self.latest_tts_audio_path = None

    def record_confirmed_message(self, speaker: str, confirmed_text: str):
        """Records an explicitly confirmed message."""
        clean = confirmed_text.strip()
        if not clean:
            return
        self.is_confirmed = True
        self.confirmed_message = clean
        self.proposed_message = clean

        ts = datetime.datetime.now().strftime("%H:%M:%S")

        log_item = ConversationLogItem(
            speaker=speaker,
            original_transcript=self.latest_raw_transcript,
            interpreted_message=self.latest_interpreted_message,
            confirmed_message=clean,
            modalities_used=list(self.used_modalities) if self.used_modalities else ["Speech"],
            was_spoken=False,
            timestamp=ts,
        )
        self.conversation_log.append(log_item)
        if len(self.conversation_log) > 30:
            self.conversation_log.pop(0)

    def mark_spoken(self):
        """Marks the latest confirmed log item as successfully spoken."""
        if self.conversation_log:
            self.conversation_log[-1].was_spoken = True

    def get_history_for_prompt(self) -> List[Dict[str, str]]:
        """Returns simplified confirmed dialogue turns for LLM reasoning prompt."""
        history = []
        for item in self.conversation_log[-8:]:
            history.append({
                "speaker": item.speaker,
                "text": item.confirmed_message,
            })
        return history

    def format_log_display(self) -> str:
        """Renders the chronological conversation log for display."""
        if not self.conversation_log:
            return "No messages confirmed in this session yet."

        lines = []
        for idx, item in enumerate(self.conversation_log, start=1):
            spoken_tag = "🔊 Spoken" if item.was_spoken else "📝 Confirmed (Not Spoken)"
            orig_tag = f" (Original: \"{item.original_transcript}\")" if item.original_transcript else ""
            mods = f" [Modalities: {', '.join(item.modalities_used)}]" if item.modalities_used else ""
            lines.append(
                f"{idx}. [{item.timestamp}] [{item.speaker}]: \"{item.confirmed_message}\" [{spoken_tag}]{orig_tag}{mods}"
            )
        return "\n\n".join(lines)

    def add_calibration_sample(self, gesture_name: str, meaning: str, feature_vector: Any):
        """Store a real captured sample for a patient gesture."""
        if gesture_name not in self.calibration_samples_store:
            self.calibration_samples_store[gesture_name] = []
            self.calibrated_gestures[gesture_name] = {
                "name": gesture_name,
                "meaning": meaning,
            }
        self.calibration_samples_store[gesture_name].append(feature_vector.copy())

    def remove_calibrated_gesture(self, gesture_name: str):
        """Remove a calibrated gesture from session state."""
        self.calibrated_gestures.pop(gesture_name, None)
        self.calibration_samples_store.pop(gesture_name, None)

    def clear_all_gestures(self):
        """Remove all calibrated gestures."""
        self.calibrated_gestures.clear()
        self.calibration_samples_store.clear()
        self.latest_recognized_gesture = "No calibrated gestures"
        self.latest_recognized_meaning = None
        self.latest_gesture_confidence = 0.0

    def format_gestures_display(self) -> str:
        """Format the configured gestures list for UI display."""
        if not self.calibrated_gestures:
            return "No patient-specific gestures calibrated in this session yet."

        lines = []
        for idx, (name, meta) in enumerate(self.calibrated_gestures.items(), start=1):
            samples = self.calibration_samples_store.get(name, [])
            count = len(samples)
            meaning = meta.get("meaning", "None")
            lines.append(f"{idx}. Gesture: \"{name}\" ➔ Meaning: \"{meaning}\" ({count} real calibration samples)")
        return "\n\n".join(lines)

    def clear(self):
        """Resets all session dialogue and ephemeral data."""
        self.latest_patient_transcript = ""
        self.latest_partner_transcript = ""
        self.latest_raw_transcript = ""
        self.audio_recorded_bytes = None
        self.latest_interpreted_message = ""
        self.alternative_interpretations.clear()
        self.reasoning_summary = ""
        self.used_modalities.clear()
        self.uncertainty_notes = None
        self.requires_clarification = False
        self.clarification_question = None
        self.proposed_message = ""
        self.is_confirmed = False
        self.confirmed_message = ""
        self.latest_tts_audio_path = None
        self.conversation_log.clear()
        self.clear_all_gestures()
        self.speech_freshness = ModalityFreshness()
        self.vision_freshness = ModalityFreshness()
        self.gesture_freshness = ModalityFreshness()
