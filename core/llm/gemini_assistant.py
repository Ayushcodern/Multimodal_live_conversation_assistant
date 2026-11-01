"""
Gemini Multimodal Intent Understanding and Reasoning Module.

Uses Google GenAI SDK (`google-genai`) to interpret spoken transcripts together with
physical vision features (hand landmarks, head movements, and conversation context).
Enforces structured Pydantic output, transparent error handling, and zero mock responses.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load API keys securely from .env and api.env
load_dotenv(".env")
load_dotenv("api.env")

# Stable and officially recommended model ID for low-latency reasoning
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class CandidateInterpretation(BaseModel):
    """Candidate message interpretation with supporting evidence."""
    message: str = Field(description="Interpreted phrase or sentence.")
    evidence_explanation: str = Field(description="Evidence supporting this candidate interpretation.")


class IntentInterpretationResponse(BaseModel):
    """Structured intent interpretation output from Gemini."""
    original_transcript: str = Field(description="Exact original speech transcript without modification.")
    main_interpretation: str = Field(description="The primary most likely interpretation.")
    candidates: List[CandidateInterpretation] = Field(
        default_factory=list,
        description="Up to 3 alternative candidate messages (fewer if only one reasonable interpretation exists)."
    )
    is_ambiguous: bool = Field(description="Whether the multimodal evidence is ambiguous or incomplete.")
    clarification_question: Optional[str] = Field(
        default=None,
        description="A targeted clarification question if ambiguity exists, else None."
    )
    missing_or_conflicting_inputs: Optional[str] = Field(
        default=None,
        description="Details on any missing, weak, or conflicting sensor signals."
    )


class GeminiAssistant:
    """Manages communication with the Google Gemini API for intent reasoning."""

    def __init__(self, model_id: str = DEFAULT_GEMINI_MODEL):
        self.model_id = model_id
        self._api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the google.genai Client."""
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key or not self._api_key.strip():
            self._client = None
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key.strip())
        except Exception:
            self._client = None

    @property
    def is_configured(self) -> bool:
        """Return True if Gemini API key is configured."""
        return self._client is not None

    def interpret_intent(
        self,
        transcript: str,
        speaker: str,
        hand_info: Optional[List[Dict[str, Any]]] = None,
        head_movement_data: Optional[Dict[str, Any]] = None,
        face_status: str = "Camera unavailable",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[bool, Optional[IntentInterpretationResponse], str]:
        """
        Send genuine multimodal context to Gemini for intent interpretation.

        Returns:
            Tuple[bool, Optional[IntentInterpretationResponse], str]:
                (success, parsed_response, status_message)
        """
        if not transcript or not transcript.strip():
            return False, None, "Input Error: Transcript is empty. Please record and transcribe speech first."

        if self._client is None:
            self._initialize_client()

        if self._client is None:
            return (
                False,
                None,
                "Configuration Error: GEMINI_API_KEY is not configured in .env or api.env. "
                "Please configure a valid Gemini API key.",
            )

        # Format sensor observations safely without speculating or fabricating data
        vision_summary = []
        if hand_info:
            hand_summary = f"{len(hand_info)} hand(s) in view."
            for h in hand_info:
                cx, cy = h.get("centroid", (0, 0))
                hand_summary += f" [Hand {h.get('hand_index')}: centroid=({cx}, {cy})]"
            vision_summary.append(hand_summary)
        else:
            vision_summary.append("No hands visible in camera.")

        if face_status == "Face detected" and head_movement_data:
            dir_str = head_movement_data.get("details", "Steady")
            dx = head_movement_data.get("delta_x", 0.0)
            dy = head_movement_data.get("delta_y", 0.0)
            vision_summary.append(f"Face visible. Head movement signal: {dir_str} (dx: {dx}px, dy: {dy}px).")
        elif face_status == "Face detected":
            vision_summary.append("Face visible. Head motion is steady.")
        else:
            vision_summary.append(f"Face tracking: {face_status}.")

        history_lines = []
        if conversation_history:
            for item in conversation_history[-6:]:  # Keep recent session context
                spk = item.get("speaker", "Unknown")
                text = item.get("text", "")
                history_lines.append(f"- {spk}: {text}")
        history_text = "\n".join(history_lines) if history_lines else "None (Beginning of session)"

        system_instruction = (
            "You are an assistive conversational AI for PS-16, helping individuals with speech impairments.\n"
            "Your task is to interpret the user's intended message by combining the verbatim speech transcript "
            "with physical vision sensor cues (hand centroid presence, head movement signals) and recent conversation context.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "1. Preserve original verbatim transcripts without silently altering words, including Hinglish/Roman Hindi words.\n"
            "2. Never claim landmark detection represents universal sign language or lip-reading.\n"
            "3. Treat head nod/shake strictly as experimental physical cues, not clinical or definitive consent.\n"
            "4. If evidence is ambiguous, incomplete, or conflicting, set is_ambiguous=True and formulate a gentle, specific clarification_question.\n"
            "5. Provide up to 3 candidate messages only if distinct reasonable interpretations exist. Do not force three duplicates.\n"
            "6. Output MUST strictly conform to the provided JSON schema."
        )

        user_prompt = (
            f"CURRENT OBSERVATION:\n"
            f"- Speaker: {speaker}\n"
            f"- Original Speech Transcript: \"{transcript.strip()}\"\n"
            f"- Vision Sensor State: {'; '.join(vision_summary)}\n\n"
            f"RECENT CONVERSATION HISTORY:\n"
            f"{history_text}\n\n"
            f"Analyze the intent and provide the structured interpretation."
        )

        # First attempt: Google GenAI SDK GenerateContent with native Pydantic schema validation
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=IntentInterpretationResponse,
            )

            response = self._client.models.generate_content(
                model=self.model_id,
                contents=user_prompt,
                config=config,
            )

            raw_text = response.text
            if not raw_text:
                return False, None, "Gemini API returned an empty response."

            parsed = IntentInterpretationResponse.model_validate_json(raw_text)
            return True, parsed, f"Intent interpreted successfully via Gemini ({self.model_id})."

        except Exception as e:
            err_str = str(e)

            # Check if generate_content failed due to model endpoint change, try Interactions API
            if hasattr(self._client, "interactions") and ("404" in err_str or "NOT_FOUND" in err_str):
                try:
                    schema_dict = IntentInterpretationResponse.model_json_schema()
                    interaction = self._client.interactions.create(
                        model=self.model_id,
                        input=f"{system_instruction}\n\n{user_prompt}",
                        response_format={"type": "object", "schema": schema_dict},
                    )
                    # Extract interaction output content
                    output_text = None
                    if hasattr(interaction, "outputs") and interaction.outputs:
                        for out in interaction.outputs:
                            if hasattr(out, "text") and out.text:
                                output_text = out.text
                                break
                    if output_text:
                        parsed = IntentInterpretationResponse.model_validate_json(output_text)
                        return True, parsed, f"Intent interpreted successfully via Gemini Interactions API ({self.model_id})."
                except Exception as inner_e:
                    err_str = f"{err_str} | Interactions fallback error: {str(inner_e)}"

            # Transparent, diagnostic error reporting
            if "403" in err_str or "PERMISSION_DENIED" in err_str or "permission_denied" in err_str:
                msg = (
                    f"Gemini API Error (403 Permission Denied): {err_str}\n\n"
                    "Google has updated API project tier policies. If your project has not enabled billing at "
                    "https://aistudio.google.com/billing, update your API key in api.env or .env to a project with active tier access."
                )
            elif "404" in err_str or "NOT_FOUND" in err_str:
                msg = f"Gemini API Error (404 Not Found): Model '{self.model_id}' is unavailable. {err_str}"
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                msg = f"Gemini API Error (429 Quota Exceeded): Rate limit or quota exhausted. {err_str}"
            else:
                msg = f"Gemini API Error: {err_str}"

            return False, None, msg
