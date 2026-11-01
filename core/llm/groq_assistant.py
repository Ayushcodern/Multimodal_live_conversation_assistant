"""
Groq Multimodal Intent Understanding and Response Refinement Module.
Phase 10: Structured Multimodal Fusion, Explicit Modality Freshness, and Ambiguity Gating.

Uses Groq's high-speed OpenAI-compatible API to interpret communication using:
- Real ElevenLabs speech transcripts (preserving English, Hindi, and Roman Hinglish)
- Explicit speaker roles (Patient vs. Conversation Partner)
- Calibrated patient gestures (if recognized)
- MediaPipe hand and head movement signals (with explicit staleness checks)
- Chronological confirmed conversation context

Returns a structured Pydantic response:
- interpreted_message
- alternative_interpretations
- language
- reasoning_summary
- used_modalities
- uncertainty_notes
- requires_clarification
- clarification_question
"""

import json
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import httpx
from pydantic import BaseModel, Field
from config.settings import get_env_var, load_project_env

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class IntentCategory(str, Enum):
    """Stable intent IDs for PS-16 communication."""
    REQUEST_WATER = "request_water"
    REQUEST_FOOD = "request_food"
    REQUEST_MEDICINE = "request_medicine"
    PAIN = "pain"
    TOILET = "toilet"
    HELP_EMERGENCY = "help_emergency"
    CALL_PERSON = "call_person"
    TIRED_REST = "tired_rest"
    TEMPERATURE = "temperature"
    REQUEST_OBJECT = "request_object"
    ENTERTAINMENT = "entertainment"
    YES = "yes"
    NO = "no"
    THANKS = "thanks"
    GREETING = "greeting"
    REPEAT_PLEASE = "repeat_please"
    WAIT = "wait"
    UNCLEAR_RETRY = "unclear_retry"
    OTHER = "other"


class CandidateMessage(BaseModel):
    """A ranked communication candidate generated for the patient."""
    rank: int = Field(
        ge=1, le=4,
        description="Rank position from 1 (most plausible) to 4 (least plausible)."
    )
    text: str = Field(
        description="First-person message in the patient's language and script."
    )
    intent: str = Field(
        default=IntentCategory.OTHER.value,
        description="Stable intent ID (e.g. request_water, pain, yes, no, unclear_retry, other)."
    )
    language: str = Field(
        default="English",
        description="Language and script (e.g. English, Roman Hindi/Hinglish, Hindi)."
    )
    why: str = Field(
        description="Brief explanation based solely on available evidence."
    )
    uncertainty_note: Optional[str] = Field(
        default=None,
        description="Ambiguity explanation if this candidate carries uncertainty."
    )


class ReplySuggestionCategory(str, Enum):
    """Four standardized categories for other-person reply suggestions."""
    YES = "yes"
    NO = "no"
    SPECIFIC = "specific"
    CLARIFICATION = "clarification"


class ReplySuggestion(BaseModel):
    """A suggested reply candidate for the patient in response to the conversation partner."""
    text: str = Field(description="First-person reply in patient's language and script.")
    intent: str = Field(description="Stable intent ID (e.g. yes, no, request_water, repeat_please, etc.).")
    language: str = Field(default="English", description="Detected language / script.")
    category: str = Field(description="Category: 'yes', 'no', 'specific', or 'clarification'.")


class ReplySuggestionsResponse(BaseModel):
    """Structured response from Groq containing exactly four categorized reply suggestions."""
    suggestions: List[ReplySuggestion] = Field(
        default_factory=list,
        description="Exactly four categorized reply suggestions (yes, no, specific, clarification)."
    )
    source_transcript: str = Field(description="The other person's verbatim speech that prompted these replies.")
    language: str = Field(default="English", description="Language of the conversation.")
    uncertainty_notes: Optional[str] = Field(default=None, description="Ambiguity explanation if applicable.")
    fallback_used: bool = Field(default=False, description="True if safe contextual reply could not be generated.")


class RefinedIntentResponse(BaseModel):
    """Refined structured intent interpretation response with ranked candidates from Groq."""
    interpreted_message: str = Field(
        description="The primary, rank-1, short, natural, patient-intended message."
    )
    candidates: List[CandidateMessage] = Field(
        default_factory=list,
        description="3 to 4 ranked candidate messages ordered from most plausible (rank 1) to least plausible."
    )
    alternative_interpretations: List[str] = Field(
        default_factory=list,
        description="Alternative candidate message strings for backward compatibility."
    )
    language: str = Field(
        default="English",
        description="Detected spoken language and script (e.g. English, Roman Hindi/Hinglish, Hindi)."
    )
    reasoning_summary: str = Field(
        description="Concise explanation of how observed speech, gestures, and sensor signals led to this interpretation."
    )
    used_modalities: List[str] = Field(
        default_factory=list,
        description="List of sensor modalities that actually influenced the interpretation (e.g. ['Speech', 'Hand Landmark', 'Calibrated Gesture'])."
    )
    uncertainty_notes: Optional[str] = Field(
        default=None,
        description="Notes on missing, conflicting, or stale sensor inputs, or why ambiguity arose."
    )
    requires_clarification: bool = Field(
        default=False,
        description="True if inputs are conflicting, highly ambiguous, or insufficient for confident interpretation."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="A gentle, specific clarification question when requires_clarification is True."
    )


class GroqAssistant:
    """Manages communication with the Groq API for multimodal intent reasoning."""

    def __init__(self, model_id: str = DEFAULT_GROQ_MODEL, timeout_seconds: float = 30.0):
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    def _get_api_key(self) -> Optional[str]:
        """Securely retrieves the GROQ_API_KEY from environment."""
        load_project_env()
        return get_env_var("GROQ_API_KEY")

    @property
    def is_configured(self) -> bool:
        """Return True if GROQ_API_KEY is configured."""
        return self._get_api_key() is not None

    def interpret_multimodal_intent(
        self,
        transcript: str,
        speaker: str,
        hand_info: Optional[List[Dict[str, Any]]] = None,
        head_movement_data: Optional[Dict[str, Any]] = None,
        face_status: str = "Camera unavailable",
        is_vision_stale: bool = True,
        calibrated_gesture_name: Optional[str] = None,
        calibrated_gesture_meaning: Optional[str] = None,
        gesture_status: str = "No calibrated gestures",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[RefinedIntentResponse], str]:
        """
        Fuses real multimodal context and queries Groq for structured intent reasoning.

        Returns:
            Tuple[bool, Optional[RefinedIntentResponse], str]:
                (success, parsed_response, status_message)
        """
        if not transcript or not transcript.strip():
            return False, None, "Input Error: Transcript is empty. Please record and transcribe speech first."

        api_key = self._get_api_key()
        if not api_key:
            return (
                False,
                None,
                "Configuration Error: GROQ_API_KEY is not configured in .env or api.env. "
                "Please configure a valid Groq API key.",
            )

        # Build transparent modality status report
        modality_reports = []

        # 1. Speech modality
        modality_reports.append(f"1. Spoken Transcript (Fresh): \"{transcript.strip()}\" [Speaker: {speaker}]")

        # 2. Camera & Vision features
        if is_vision_stale or face_status == "Camera unavailable":
            modality_reports.append("2. Camera & Vision Stream: UNAVAILABLE or STALE. (Visual cues must not be inferred).")
        else:
            v_cues = []
            if hand_info:
                v_cues.append(f"{len(hand_info)} hand(s) detected in camera view")
            else:
                v_cues.append("No hands visible in camera")

            if face_status == "Face detected" and head_movement_data:
                dir_str = head_movement_data.get("details", "Steady")
                v_cues.append(f"Head movement cue (Experimental): {dir_str}")
            elif face_status == "Face detected":
                v_cues.append("Face visible, head steady")
            else:
                v_cues.append(f"Face tracking: {face_status}")

            modality_reports.append(f"2. Camera & Vision Stream (Live): {'; '.join(v_cues)}")

        # 3. Patient Calibrated Gesture modality
        if not is_vision_stale and calibrated_gesture_name and calibrated_gesture_meaning and "Recognized" in gesture_status:
            modality_reports.append(
                f"3. Patient Calibrated Gesture (Live Match): \"{calibrated_gesture_name}\" "
                f"➔ Assigned meaning: \"{calibrated_gesture_meaning}\""
            )
        elif not is_vision_stale and "Uncertain" in gesture_status:
            modality_reports.append("3. Patient Calibrated Gesture: UNCERTAIN / UNCLASSIFIED motion (Do not guess meaning).")
        else:
            modality_reports.append(f"3. Patient Calibrated Gesture: {gesture_status}.")

        # 4. Confirmed conversation history
        history_lines = []
        if conversation_history:
            for item in conversation_history[-8:]:
                spk = item.get("speaker", "Unknown")
                text = item.get("text", "")
                history_lines.append(f"- [{spk}]: \"{text}\"")
        history_text = "\n".join(history_lines) if history_lines else "None (Session started)"

        # 5. Patient Profile (Milestone 3)
        profile_text = "No profile context active."
        if patient_profile and patient_profile.get("enabled", False):
            p_parts = []
            if patient_profile.get("display_name"):
                p_parts.append(f"Name: {patient_profile['display_name']}")
            if patient_profile.get("preferred_language") and patient_profile["preferred_language"] != "Auto":
                p_parts.append(f"Preferred Language: {patient_profile['preferred_language']}")
            if patient_profile.get("preferred_script"):
                p_parts.append(f"Preferred Script: {patient_profile['preferred_script']}")
            if patient_profile.get("common_needs"):
                p_parts.append(f"Common Needs: {', '.join(patient_profile['common_needs'])}")
            if patient_profile.get("frequently_used_phrases"):
                p_parts.append(f"Frequent Phrases: {', '.join(patient_profile['frequently_used_phrases'])}")
            if patient_profile.get("communication_notes"):
                p_parts.append(f"Notes: {patient_profile['communication_notes']}")
            profile_text = "; ".join(p_parts) if p_parts else "Default profile preferences active."

        schema_json = json.dumps(RefinedIntentResponse.model_json_schema(), indent=2)

        system_instruction = (
            "You are an assistive conversational AI for PS-16, dedicated to helping individuals with speech impairments.\n"
            "Your task is to interpret the user's intended message by combining the verbatim speech transcript with available "
            "physical vision sensor cues, patient-calibrated gestures, and recent conversation context.\n\n"
            "CRITICAL RULES:\n"
            "1. LANGUAGE PRESERVATION: Preserve original words exactly. Keep Hindi, English, and Roman Hindi / Hinglish in Roman script. "
            "Never automatically translate Roman Hindi into Devanagari. Keep the interpreted message concise, natural, and first-person.\n"
            "2. 3 TO 4 RANKED CANDIDATES: You MUST produce exactly 3 to 4 candidates ordered from rank 1 (most plausible) to rank 3 or 4.\n"
            "   - 'interpreted_message' MUST match candidate rank 1's text.\n"
            "   - 'intent' MUST be one of the stable intent IDs: request_water, request_food, request_medicine, pain, toilet, help_emergency, "
            "call_person, tired_rest, temperature, request_object, entertainment, yes, no, thanks, greeting, repeat_please, wait, unclear_retry, other.\n"
            "   - If evidence is weak, muffled, or ambiguous, include 'unclear_retry' as a candidate rather than inventing facts.\n"
            "   - 'why' must strictly state which evidence (speech words, calibrated gesture, or conversation context) supports that specific candidate.\n"
            "3. PATIENT PROFILE USAGE: If a patient profile is provided, use it only as gentle context for vocabulary and preferences. "
            "NEVER let profile preferences override contradictory current speech or gestures. A preference for water does not mean the user wants water now.\n"
            "4. GESTURE INTERPRETATION: If a patient-calibrated gesture is recognized, incorporate its assigned meaning as supporting evidence. "
            "If gestures are uncalibrated, uncertain, or unavailable, DO NOT guess or fabricate gesture meanings.\n"
            "5. NO CONSENT BY MOTION: Never interpret head movement (nod/shake) or facial expressions as legal, medical, or definitive consent.\n"
            "6. MISSING SENSORS: Missing or stale camera streams are unavailable, not negative responses.\n"
            "7. AMBIGUITY HANDLING: If speech is muffled or sensor signals conflict, set requires_clarification=True, identify uncertainty in "
            "uncertainty_notes, and formulate a gentle, targeted clarification_question.\n"
            "8. NO FAKE PROBABILITIES: Do not represent LLM confidence as a calibrated statistical probability. Ranks are relative ordering only.\n"
            "9. STRICT VALID JSON ONLY: Return ONLY standard raw JSON matching the schema. Do NOT use markdown bolding (e.g. **key**), markdown codeblocks (```), or comments inside or around the JSON object.\n"
            f"{schema_json}"
        )

        user_prompt = (
            f"OBSERVED SENSOR MODALITIES:\n"
            + "\n".join(modality_reports)
            + f"\n\nPATIENT PROFILE PREFERENCES (Contextual only):\n{profile_text}\n\n"
            f"SESSION CONVERSATION LOG (Confirmed dialogue turns):\n{history_text}\n\n"
            f"Analyze the multimodal evidence and return the structured JSON interpretation."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(GROQ_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                resp_json = response.json()
                choices = resp_json.get("choices", [])
                if not choices:
                    return False, None, "Groq API returned an empty choices list."

                raw_content = choices[0].get("message", {}).get("content", "")
                if not raw_content:
                    return False, None, "Groq API returned an empty message content."

                parsed = RefinedIntentResponse.model_validate_json(raw_content)

                # Ensure candidates are populated and sorted
                if parsed.candidates:
                    parsed.candidates.sort(key=lambda c: c.rank)
                    # Sync interpreted_message with rank 1
                    parsed.interpreted_message = parsed.candidates[0].text
                    # Sync alternative_interpretations
                    parsed.alternative_interpretations = [c.text for c in parsed.candidates[1:]]
                elif parsed.interpreted_message:
                    # Fallback single candidate wrapper if model emitted legacy format
                    parsed.candidates = [
                        CandidateMessage(
                            rank=1,
                            text=parsed.interpreted_message,
                            intent=IntentCategory.OTHER.value,
                            language=parsed.language,
                            why="Derived from observed input signals.",
                        )
                    ]
                    for i, alt in enumerate(parsed.alternative_interpretations[:3], start=2):
                        parsed.candidates.append(
                            CandidateMessage(
                                rank=i,
                                text=alt,
                                intent=IntentCategory.OTHER.value,
                                language=parsed.language,
                                why="Plausible alternative interpretation.",
                            )
                        )

                return True, parsed, f"Intent interpreted successfully via Groq ({self.model_id})."

            status_code = response.status_code
            err_body = response.text
            if status_code == 401:
                msg = "Groq API Error (401 Unauthorized): Invalid GROQ_API_KEY. Please verify key in .env or api.env."
            elif status_code == 404:
                msg = f"Groq API Error (404 Model Not Found): Model '{self.model_id}' is not available. {err_body}"
            elif status_code == 429:
                msg = f"Groq API Error (429 Rate Limit): Quota or rate limit exceeded. {err_body}"
            else:
                msg = f"Groq API Error ({status_code}): {err_body}"

            return False, None, msg

        except Exception as e:
            return False, None, f"Groq Connection Error: {str(e)}"

    def generate_reply_suggestions(
        self,
        partner_transcript: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[ReplySuggestionsResponse], str]:
        """
        Generates exactly four categorized reply suggestions for the patient
        in response to the conversation partner's speech:
        1. Yes / Agree
        2. No / Disagree
        3. Specific Contextual Answer
        4. Clarification / Repeat
        """
        api_key = self._get_api_key()
        if not api_key:
            return False, None, "GROQ_API_KEY is not configured in .env or api.env."

        if not partner_transcript or not partner_transcript.strip():
            return False, None, "Partner transcript is empty. Cannot generate reply suggestions."

        # Recent dialogue context
        history_lines = []
        if conversation_history:
            for item in conversation_history[-8:]:
                spk = item.get("speaker", "Unknown")
                text = item.get("text", "")
                history_lines.append(f"- [{spk}]: \"{text}\"")
        history_text = "\n".join(history_lines) if history_lines else "None (Beginning of conversation)"

        schema_json = json.dumps(ReplySuggestionsResponse.model_json_schema(), indent=2)

        system_instruction = (
            "You are an assistive conversational AI for PS-16 helping a patient with speech impairments respond to a conversation partner.\n"
            "The conversation partner just spoke to the patient. Generate exactly 4 categorized reply suggestions for the patient.\n\n"
            "MANDATORY CATEGORIES:\n"
            "1. 'yes': An affirmative, agreement, or confirmation response in the patient's language/script.\n"
            "2. 'no': A polite negative, disagreement, or decline response.\n"
            "3. 'specific': A direct, relevant, contextual answer to what the partner asked or stated.\n"
            "4. 'clarification': A gentle request to repeat or clarify (e.g. 'Please repeat', 'Kya kaha aapne?').\n\n"
            "CRITICAL RULES:\n"
            "1. LANGUAGE PRESERVATION: Match the spoken language and script of the conversation. Keep Roman Hindi/Hinglish in Roman script. Never translate Roman Hindi into Devanagari.\n"
            "2. PATIENT PERSPECTIVE: All replies must be concise, natural, first-person patient utterances (e.g. 'Haan, bilkul', 'I don't think so', 'Please wait').\n"
            "3. NO INVENTED FACTS: Do not invent medical claims or facts not in the dialogue.\n"
            "4. STRICT VALID JSON ONLY: Return ONLY standard raw JSON matching the schema without markdown bolding, codeblocks, or extra text.\n"
            f"{schema_json}"
        )

        user_prompt = (
            f"CONVERSATION PARTNER'S UTTERANCE:\n\"{partner_transcript}\"\n\n"
            f"PATIENT PROFILE CONTEXT (Common needs & phrases):\n"
            f"{json.dumps(patient_profile) if patient_profile and patient_profile.get('enabled') else 'None'}\n\n"
            f"RECENT DIALOGUE TURNS:\n{history_text}\n\n"
            f"Generate the 4 categorized reply suggestions in valid JSON."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(GROQ_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                resp_json = response.json()
                choices = resp_json.get("choices", [])
                if not choices:
                    return False, None, "Groq API returned an empty choices list."

                raw_content = choices[0].get("message", {}).get("content", "")
                if not raw_content:
                    return False, None, "Groq API returned an empty message content."

                parsed = ReplySuggestionsResponse.model_validate_json(raw_content)

                # Guarantee fallback category distribution if model missed any
                categories_present = {s.category.lower() for s in parsed.suggestions}
                needed = [("yes", "Yes, that's right", "yes"),
                          ("no", "No, not right now", "no"),
                          ("specific", "I understand", "other"),
                          ("clarification", "Please repeat that", "repeat_please")]
                for cat, default_text, default_intent in needed:
                    if cat not in categories_present and len(parsed.suggestions) < 4:
                        parsed.suggestions.append(
                            ReplySuggestion(
                                text=default_text,
                                intent=default_intent,
                                language=parsed.language,
                                category=cat,
                            )
                        )
                return True, parsed, f"Generated {len(parsed.suggestions)} reply suggestions via Groq."

            return False, None, f"Groq Error ({response.status_code}): {response.text}"

        except Exception as e:
            return False, None, f"Groq Connection Error: {str(e)}"
