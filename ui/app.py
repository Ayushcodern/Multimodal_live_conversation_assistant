"""
User Interface Module for PS-16 Multimodal Live Conversation Assistant.
Phase 10: Accessible Response Refinement, Multimodal Fusion & Confirmation Safety.

Features:
- Live Vision: OpenCV + MediaPipe Hand and Face/Head tracking.
- Patient Gesture Calibration: Real-time camera preview & calibration inside Tab 2.
- Audio Capture: Windows Default Microphone auto-detection + ElevenLabs Scribe STT.
- Multimodal AI Reasoning: Groq (openai/gpt-oss-120b) fusing speech, vision, calibrated gestures, and history.
- Accessible Response Interface: Clear original transcript, proposed message, alternatives, used modalities, and uncertainty warnings.
- Strict Confirmation Safeguards: Any edit, rejection, re-interpretation, or input change invalidates confirmation.
- Text-to-Speech: ElevenLabs Flash TTS with verified free-tier premade voice.
- Shared Session State: Isolated per-client Gradio session state preventing cross-talk.
"""

from typing import Optional, Tuple, Any, List, Dict
import gradio as gr
import numpy as np
import tempfile
import time

from core.vision.camera import CameraManager
from core.vision.gesture_detector import HandGestureDetector
from core.vision.face_detector import FaceHeadDetector
from core.vision.gesture_classifier import PatientGestureClassifier
from core.audio.recorder import AudioRecorder
from core.audio.scribe_stt import ScribeTranscriber
from core.audio.flash_tts import FlashTTS
from core.llm.groq_assistant import GroqAssistant, RefinedIntentResponse
from core.session_state import SessionState


def create_app(
    camera_manager: Optional[CameraManager] = None,
    gesture_detector: Optional[HandGestureDetector] = None,
    face_detector: Optional[FaceHeadDetector] = None,
    gesture_classifier: Optional[PatientGestureClassifier] = None,
    audio_recorder: Optional[AudioRecorder] = None,
    scribe_transcriber: Optional[ScribeTranscriber] = None,
    groq_assistant: Optional[GroqAssistant] = None,
    flash_tts: Optional[FlashTTS] = None,
) -> gr.Blocks:
    """
    Build and configure the Gradio application block for full multimodal live assistance.
    """
    if camera_manager is None:
        camera_manager = CameraManager(camera_index=0)
    if gesture_detector is None:
        gesture_detector = HandGestureDetector()
    if face_detector is None:
        face_detector = FaceHeadDetector()
    if gesture_classifier is None:
        gesture_classifier = PatientGestureClassifier(similarity_threshold=0.88, temporal_window=5, min_stable_frames=3)
    if audio_recorder is None:
        audio_recorder = AudioRecorder(sample_rate=16000, channels=1)
    if scribe_transcriber is None:
        scribe_transcriber = ScribeTranscriber()
    if groq_assistant is None:
        groq_assistant = GroqAssistant()
    if flash_tts is None:
        flash_tts = FlashTTS()

    # Query initial Windows default microphone
    _, _, default_mic_name, default_mic_status = AudioRecorder.get_default_input_device()

    # Accessible Custom CSS for PS-16 Visual Foundation
    custom_css = """
    /* Deep Charcoal Surface Theme with Accessible Contrast */
    :root {
        --bg-deep: #121214;
        --surface-card: #18181b;
        --surface-border: #27272a;
        --primary-orange: #ea580c;
        --primary-hover: #f97316;
        --text-high: #f4f4f5;
        --text-muted: #a1a1aa;
        --accent-green: #16a34a;
        --accent-red: #dc2626;
    }

    body, .gradio-container {
        background-color: var(--bg-deep) !important;
        color: var(--text-high) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Accessible Target Sizes: Buttons min 44px height */
    button.lg, button.sm, button.primary, button.secondary, .gr-button {
        min-height: 44px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Primary Accent */
    button.primary, .gr-button-primary {
        background: var(--primary-orange) !important;
        border: 1px solid var(--primary-orange) !important;
        color: #ffffff !important;
    }
    button.primary:hover, .gr-button-primary:hover {
        background: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
    }

    /* Secondary Neutral */
    button.secondary, .gr-button-secondary {
        background: #27272a !important;
        border: 1px solid #3f3f46 !important;
        color: var(--text-high) !important;
    }
    button.secondary:hover, .gr-button-secondary:hover {
        background: #3f3f46 !important;
    }

    /* Destructive / Urgent */
    button.stop, .gr-button-stop {
        background: #7f1d1d !important;
        border: 1px solid #b91c1c !important;
        color: #fef2f2 !important;
    }
    button.stop:hover, .gr-button-stop:hover {
        background: var(--accent-red) !important;
    }

    /* Clear Focus Outlines for Keyboard Accessibility */
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {
        outline: 3px solid var(--primary-hover) !important;
        outline-offset: 2px !important;
    }

    /* Card Panels & Textboxes */
    .gr-panel, .gr-box, .gr-form, .gr-group {
        background-color: var(--surface-card) !important;
        border: 1px solid var(--surface-border) !important;
        border-radius: 10px !important;
    }

    /* High-Contrast Input Text */
    input, textarea {
        color: #ffffff !important;
        background-color: #202024 !important;
        border-color: #3f3f46 !important;
    }

    /* Tab Navigation Styling */
    .tab-nav button {
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
    }
    .tab-nav button.selected {
        border-bottom: 3px solid var(--primary-orange) !important;
        color: var(--primary-orange) !important;
    }
    """

    with gr.Blocks(title="Sahara: Assistive Communication", css=custom_css) as demo:
        # Isolated per-browser-session state
        session_state = gr.State(delete_callback=None, value=SessionState)

        gr.Markdown(
            """
            # Sahara
            ### Multimodal Assistive Communication
            """
        )

        with gr.Accordion("ℹ️ Instructions & Live Hackathon Demonstration Guide", open=False):
            gr.Markdown(
                """
                **Step-by-Step Demonstration Flow:**
                1. **Vision (Tab 1):** Click **Start Camera** to activate live OpenCV video and MediaPipe Hand/Face landmark tracking.
                2. **Gesture Calibration (Tab 2):** Define a custom gesture name & message (e.g. *Water Request*), hold your hand in view, and capture 3–5 samples to save a personalized gesture.
                3. **Speech Input (Tab 3):** Select the speaker (*Patient* or *Conversation Partner*), click **Start Recording**, speak into the Windows default microphone, and click **Transcribe with ElevenLabs Scribe**.
                4. **AI Reasoning & Confirmation (Tab 4):** Click **Analyze Intent with Groq** to fuse speech, gesture, and conversation context. Review the proposed message.
                5. **Speech Synthesis (Tab 4):** Click **Confirm Message** to unlock the ElevenLabs Flash TTS engine, then click **Speak Message** to synthesize and play audio.
                
                *Safety Note: All TTS synthesis is strictly locked until explicit patient confirmation. Editing or rejecting proposed text immediately invalidates confirmation.*
                """
            )

        # Milestone 9: High-Priority Emergency / Quick-Help Pathway Banner
        with gr.Accordion("🚨 Quick Caregiver Alert & Urgent Help (Safety Gated)", open=False):
            gr.Markdown(
                "**⚠️ ASSISTIVE COMMUNICATION NOTICE**: This tool aids in-room spoken communication with caregivers and family. "
                "It does **NOT** automatically dial emergency services (911/112) or dispatch first responders. "
                "In a medical emergency, caregivers should directly call local emergency services."
            )
            with gr.Row():
                urg_btn_help = gr.Button("🚨 I need urgent help!", variant="stop", scale=1)
                urg_btn_pain = gr.Button("⚠️ Severe pain, please assist me.", variant="secondary", scale=1)
                urg_btn_caregiver = gr.Button("📞 Call my caregiver immediately.", variant="secondary", scale=1)
                urg_btn_breathe = gr.Button("🫁 Trouble breathing / dizzy.", variant="secondary", scale=1)
                urg_btn_cancel = gr.Button("🛑 Cancel / Clear Urgent Alert", variant="secondary", scale=1)

        # Hidden status components for backend callback compatibility
        cam_pill = gr.Textbox(label="Camera", value="Off", interactive=False, visible=False)
        mic_pill = gr.Textbox(label="Microphone", value="Ready", interactive=False, visible=False)
        confirm_pill = gr.Textbox(label="Voice Output", value="Locked", interactive=False, visible=False)

        # 5 User-Centred Sections (Blueprint Milestone 2)
        with gr.Tabs() as main_nav_tabs:
            # =========================================================================
            # SECTION 1: 🏠 Home / Live Communication (Landing Page)
            # =========================================================================
            with gr.Tab("🏠 Home", id="tab_home"):
                gr.Markdown("### How would you like to communicate?")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown(
                                """
                                #### ⚡ Instant Quick Phrases
                                Choose a ready-made phrase and speak it immediately with ElevenLabs TTS without needing camera or AI processing.
                                """
                            )
                            go_quick_btn = gr.Button("Open Quick Phrases ➔", variant="primary")

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown(
                                """
                                #### 🧠 Speak with AI
                                Hold natural, multimodal conversations using your voice, natural gestures, and partner replies with Groq reasoning.
                                """
                            )
                            go_ai_btn = gr.Button("Open AI Conversation ➔", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### ⚡ Quick Access Shortcuts")
                with gr.Row():
                    home_quick_help = gr.Button("🚨 I need help", variant="stop")
                    home_quick_water = gr.Button("💧 I need water", variant="secondary")
                    home_quick_yes = gr.Button("✅ Yes", variant="secondary")
                    home_quick_no = gr.Button("❌ No", variant="secondary")
                    home_quick_wait = gr.Button("⏳ Please wait", variant="secondary")

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 Recent Confirmed Dialogue")
                        home_history_box = gr.Textbox(
                            label="Session History",
                            value="No messages confirmed in this session yet.",
                            lines=8,
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### ℹ️ Demonstration & Help")
                        gr.Markdown(
                            """
                            • **Tab 2 (Quick Phrases)**: Direct speech without waiting for STT or AI.
                            • **Tab 3 (Personalise)**: Configure your preferred language, needs, and train custom gestures.
                            • **Tab 4 (Voice & Speech)**: Choose voice persona and record audio.
                            • **Tab 5 (AI Understanding)**: Full multimodal fusion and safety confirmation.
                            """
                        )

            # =========================================================================
            # SECTION 2: ⚡ Quick Phrases
            # =========================================================================
            with gr.Tab("⚡ Quick Phrases", id="tab_quick"):
                gr.Markdown(
                    """
                    ### ⚡ Instant Communication Phrases
                    **Tap any message below to review it.** You always decide when it is spoken.
                    *No camera, microphone, or cloud AI required for instant phrases.*
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### 🟢 Basic Communication")
                            quick_btn_yes = gr.Button("✅ Yes.", variant="secondary")
                            quick_btn_no = gr.Button("❌ No.", variant="secondary")
                            quick_btn_wait = gr.Button("⏳ Please wait.", variant="secondary")
                            quick_btn_rep = gr.Button("🔁 Please repeat.", variant="secondary")
                            quick_btn_thx = gr.Button("🙏 Thank you.", variant="secondary")

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### 💧 Daily Needs")
                            quick_btn_water = gr.Button("💧 I need water.", variant="secondary")
                            quick_btn_food = gr.Button("🍲 I am hungry.", variant="secondary")
                            quick_btn_med = gr.Button("💊 I need my medicine.", variant="secondary")
                            quick_btn_wash = gr.Button("🚻 I need the washroom.", variant="secondary")
                            quick_btn_tired = gr.Button("🛌 I am tired.", variant="secondary")
                            with gr.Row():
                                quick_btn_hot = gr.Button("☀️ Feel hot", variant="secondary", scale=1)
                                quick_btn_cold = gr.Button("❄️ Feel cold", variant="secondary", scale=1)

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### 🆘 Urgent & Assistance")
                            quick_btn_help = gr.Button("🚨 I need help.", variant="stop")
                            quick_btn_pain = gr.Button("⚠️ I am in pain.", variant="secondary")
                            quick_btn_call = gr.Button("📞 Call someone.", variant="secondary")
                            quick_btn_stop = gr.Button("🛑 Please stop.", variant="secondary")

                        with gr.Group():
                            gr.Markdown("#### ✏️ Custom Phrases")
                            with gr.Row():
                                custom_phrase_input = gr.Textbox(
                                    label="Add Phrase",
                                    placeholder="Type a custom phrase...",
                                    scale=3,
                                )
                                add_custom_btn = gr.Button("➕ Add", variant="primary", scale=1)
                            custom_phrase_selector = gr.Dropdown(
                                label="Your Saved Custom Phrases",
                                choices=[],
                                value=None,
                                interactive=True,
                            )
                            with gr.Row():
                                use_custom_btn = gr.Button("Select Phrase", variant="secondary", scale=1)
                                remove_custom_btn = gr.Button("🗑 Remove", variant="secondary", scale=1)
                            custom_phrases_display = gr.Textbox(
                                label="In-Session Custom Phrases",
                                value="No custom phrases added yet.",
                                lines=2,
                                interactive=False,
                            )

                # Dedicated Preview, Confirmation, and TTS controls directly on Quick Phrases
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 📝 Selected Message (Review & Edit)")
                            quick_phrase_preview = gr.Textbox(
                                label="Message Preview (Editable)",
                                placeholder="Tap any phrase button above...",
                                lines=2,
                                interactive=True,
                            )
                            with gr.Row():
                                quick_confirm_btn = gr.Button("✅ Confirm Message", variant="primary", scale=2)
                                quick_edit_btn = gr.Button("✏️ Update Text", variant="secondary", scale=1)
                                quick_cancel_btn = gr.Button("❌ Clear", variant="stop", scale=1)

                            quick_action_status = gr.Textbox(
                                label="Confirmation Status",
                                value="Awaiting selection. Voice output is strictly locked until you confirm.",
                                interactive=False,
                            )

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🔊 Speech Output (ElevenLabs Flash TTS)")
                            quick_confirmed_display = gr.Textbox(
                                label="Confirmed Message (Ready to Speak)",
                                value="",
                                placeholder="Confirm your message to unlock speech output...",
                                interactive=False,
                                lines=2,
                            )
                            quick_speak_btn = gr.Button(
                                "🗣️ Speak Confirmed Message",
                                variant="primary",
                                interactive=False,
                            )
                            quick_audio_playback = gr.Audio(
                                label="Voice Playback",
                                type="filepath",
                                interactive=False,
                                autoplay=False,
                            )
                            quick_tts_status = gr.Textbox(
                                label="Voice Status",
                                value=f"ElevenLabs Flash TTS ({flash_tts.model_id}) ready. Voice: Roger (Premade).",
                                interactive=False,
                            )

            # =========================================================================
            # SECTION 3: 👤 Personalise (Profile & Gesture Calibration)
            # =========================================================================
            with gr.Tab("👤 Personalise", id="tab_personalise"):
                gr.Markdown(
                    """
                    ### 👤 Personalise Your Communication Experience
                    Customize your communication preferences and train your own physical gestures.
                    *Preferences help the AI suggest the most relevant words without overriding your actual speech.*
                    """
                )
                with gr.Accordion("📋 Communication Profile & Preferences", open=True):
                    with gr.Row():
                        with gr.Column(scale=1):
                            profile_enable_cb = gr.Checkbox(
                                label="Enable Personal Profile Suggestions",
                                value=True,
                                info="When enabled, Groq considers your common needs and frequent vocabulary."
                            )
                            profile_name_input = gr.Textbox(
                                label="Preferred Name or Nickname",
                                placeholder="e.g. Alex",
                                value="Alex",
                            )
                            with gr.Row():
                                profile_lang_dropdown = gr.Dropdown(
                                    label="Spoken Language",
                                    choices=["Auto", "English", "Hindi", "Hinglish"],
                                    value="Auto",
                                    scale=1,
                                )
                                profile_script_dropdown = gr.Dropdown(
                                    label="Output Text Script",
                                    choices=["Roman", "Devanagari"],
                                    value="Roman",
                                    scale=1,
                                    info="Roman keeps Hinglish/Hindi in natural Roman letters."
                                )
                            profile_notes_input = gr.Textbox(
                                label="Communication Notes / Assistance Preferences",
                                placeholder="e.g. Speaks in brief 2-3 word sentences; slight tremor in hand.",
                                lines=2,
                            )

                        with gr.Column(scale=1):
                            profile_needs_input = gr.Textbox(
                                label="Common Daily Needs (Comma-separated)",
                                value="Water, Food, Medicine, Rest, Washroom",
                                lines=2,
                            )
                            profile_phrases_input = gr.Textbox(
                                label="Frequently Spoken Phrases (One per line)",
                                value="Please give me water\nI need my medicine\nPlease call someone\nI am feeling tired\nThank you",
                                lines=5,
                            )
                            with gr.Row():
                                profile_save_btn = gr.Button("💾 Save Profile", variant="primary")
                                profile_clear_btn = gr.Button("🗑 Clear Profile", variant="secondary")

                            profile_status_box = gr.Textbox(
                                label="Profile Status",
                                value="Default profile preferences active for this session.",
                                interactive=False,
                            )

                with gr.Accordion("🖐️ Personal Gesture Training (Guided 4-Step Setup)", open=True):
                    gr.Markdown(
                        "**Train your own natural physical gestures.** We extract scale-invariant geometric hand positions so you can signal requests even when speech is difficult."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("#### Step 1 & 2: Camera & Pose Setup")
                                calib_video_display = gr.Image(
                                    label="Calibration Camera Feed",
                                    streaming=True,
                                    type="numpy",
                                    interactive=False,
                                )
                                with gr.Row():
                                    calib_start_cam_btn = gr.Button("🎥 Start Camera", variant="primary")
                                    calib_stop_cam_btn = gr.Button("⏹ Stop Camera", variant="secondary")

                                calib_cam_status = gr.Textbox(
                                    label="Camera Status",
                                    value="Disconnected",
                                    interactive=False,
                                )

                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("#### Step 3 & 4: Name, Capture & Save")
                                with gr.Row():
                                    gesture_name_input = gr.Textbox(
                                        label="Gesture Name",
                                        placeholder="e.g. Thumbs Up, Open Palm",
                                        scale=1,
                                    )
                                    gesture_meaning_input = gr.Textbox(
                                        label="Intended Meaning / Phrase",
                                        placeholder="e.g. I need water, Yes, Stop",
                                        scale=1,
                                    )
                                with gr.Row():
                                    capture_sample_btn = gr.Button("📸 Step 3: Capture Sample", variant="secondary", scale=1)
                                    save_gesture_btn = gr.Button("💾 Step 4: Save Gesture", variant="primary", scale=1)

                                calib_feedback_box = gr.Textbox(
                                    label="Calibration Progress",
                                    value="Enter gesture name and meaning, start camera, and capture 3-5 samples.",
                                    interactive=False,
                                )

                            with gr.Group():
                                gr.Markdown("#### Saved Gestures Library")
                                gestures_list_box = gr.Textbox(
                                    label="Your Trained Gestures (Active in Session)",
                                    value="No gestures calibrated in this session yet.",
                                    lines=3,
                                    interactive=False,
                                )
                                with gr.Row():
                                    delete_gesture_name = gr.Textbox(
                                        label="Gesture to Delete",
                                        placeholder="Type exact gesture name...",
                                        scale=2,
                                    )
                                    delete_gesture_btn = gr.Button("🗑 Delete", variant="secondary", scale=1)
                                    clear_all_gestures_btn = gr.Button("🗑 Clear Library", variant="secondary", scale=1)

                                gesture_manage_status = gr.Textbox(
                                    label="Action Status",
                                    value="Ready.",
                                    interactive=False,
                                )
                                gesture_pill = gr.Textbox(
                                    label="Gesture Status Indicator",
                                    value="IDLE (No Gesture)",
                                    interactive=False,
                                    visible=False,
                                )

            # =========================================================================
            # SECTION 4: 🎙️ Voice & Speech (Speech Recording & Voice Persona)
            # =========================================================================
            with gr.Tab("🎙️ Voice & Speech", id="tab_voice_speech"):
                gr.Markdown(
                    """
                    ### 🎙️ Speech Recording & Voice Persona Settings
                    Record spoken speech using your microphone and choose your speech synthesis voice persona.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🎤 Speech Input (ElevenLabs Scribe)")
                            speaker_selector = gr.Radio(
                                label="Who is speaking?",
                                choices=["Patient", "Conversation Partner"],
                                value="Patient",
                                info="Select who is speaking to correctly maintain dialogue context."
                            )
                            default_mic_box = gr.Textbox(
                                label="Connected Microphone",
                                value=default_mic_name,
                                interactive=False,
                            )
                            with gr.Row():
                                record_btn = gr.Button("🎤 Start Recording", variant="primary", scale=1)
                                stop_rec_btn = gr.Button("⏹ Stop Recording", variant="stop", scale=1)

                            rec_status_box = gr.Textbox(
                                label="Recording Status",
                                value=f"Microphone ready. ({default_mic_status})",
                                interactive=False,
                            )
                            transcribe_btn = gr.Button("⚡ Transcribe Spoken Audio", variant="primary")

                            raw_transcript_box = gr.Textbox(
                                label="What we heard (Verbatim Transcript)",
                                placeholder="Recorded speech transcription will appear here...",
                                lines=3,
                                interactive=False,
                            )
                            stt_status_box = gr.Textbox(
                                label="Transcription Status",
                                value=(
                                    "ElevenLabs Scribe ready."
                                    if scribe_transcriber.is_configured
                                    else "Notice: ELEVENLABS_API_KEY is not configured in .env"
                                ),
                                interactive=False,
                            )
                            stt_pill = gr.Textbox(
                                label="Transcription State",
                                value="IDLE",
                                interactive=False,
                                visible=False,
                            )

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🔊 Voice Persona Output (ElevenLabs Flash TTS)")
                            active_voice_dropdown = gr.Dropdown(
                                label="Choose Spoken Voice Persona",
                                choices=[
                                    "Roger (Premade - Casual & Resonant) [Free-Tier Eligible]",
                                    "Sarah (Premade - Soft & Warm) [Free-Tier Eligible]",
                                    "Laura (Premade - Calm & Clear) [Free-Tier Eligible]",
                                    "Charlie (Premade - Confident) [Free-Tier Eligible]",
                                    "George (Premade - Deep & Articulate) [Free-Tier Eligible]",
                                    "Alice (Premade - Expressive) [Free-Tier Eligible]",
                                ],
                                value="Roger (Premade - Casual & Resonant) [Free-Tier Eligible]",
                                info="Verified high-speed premade voices compatible with free and paid tiers."
                            )
                            apply_voice_btn = gr.Button("💾 Apply Voice Selection", variant="primary")
                            voice_selection_status = gr.Textbox(
                                label="Current Active Persona",
                                value="Active Voice: Roger (Default Free Premade).",
                                interactive=False,
                            )

                        with gr.Accordion("🔒 Custom Voice Cloning (Entitlement-Gated)", open=False):
                            gr.Markdown(
                                "**Ethical & Account Entitlement Notice:** Voice cloning strictly requires verified written consent. "
                                "Free-tier ElevenLabs accounts do not include instant cloning privileges; verified premade voices remain 100% active."
                            )
                            with gr.Row():
                                check_cloning_btn = gr.Button("🔍 Check Account Entitlement", variant="secondary")
                            consent_feedback_box = gr.Textbox(
                                label="Entitlement & Consent Status",
                                value="Voice cloning is restricted on free-tier ElevenLabs accounts.",
                                lines=2,
                                interactive=False,
                            )
                            consent_checkbox = gr.Checkbox(
                                label="I explicitly consent to creating a digital clone of MY OWN voice.",
                                value=False,
                            )
                            consent_signer_input = gr.Textbox(
                                label="Full Legal Name / Electronic Signature",
                                placeholder="Type full name to verify...",
                            )
                            with gr.Row():
                                submit_consent_btn = gr.Button("📝 Record Consent & Request Clone", variant="stop")
                                delete_voice_btn = gr.Button("🗑 Revoke Consent & Revert to Roger", variant="secondary")

            # =========================================================================
            # SECTION 5: 🧠 AI Understanding (Multimodal Reasoning & Confirmation)
            # =========================================================================
            with gr.Tab("🧠 AI Understanding", id="tab_ai_understanding"):
                gr.Markdown(
                    """
                    ### 🧠 Multimodal AI Conversation Assistant
                    Fuses speech audio, natural gestures, and dialogue context with Groq LLM reasoning.
                    *Review the original words, select or edit proposed interpretations, and explicitly confirm before speaking.*
                    """
                )
                with gr.Row():
                    # Left column: Multimodal Reasoning & Communication Interpretation
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.Markdown("#### Step 1: Active Modalities & Reasoning")
                            observed_modalities_box = gr.Textbox(
                                label="Sensors Available for Context Reasoning",
                                value="Awaiting speech recording or gesture...",
                                interactive=False,
                                lines=2,
                            )
                            with gr.Row():
                                interpret_btn = gr.Button("🧠 Analyze Intent with Groq", variant="primary", scale=2)
                                retry_btn = gr.Button("🔄 Re-Analyze / Retry", variant="secondary", scale=1)

                            ai_status_box = gr.Textbox(
                                label="Groq Status",
                                value=(
                                    f"Groq ({groq_assistant.model_id}) ready."
                                    if groq_assistant.is_configured
                                    else "Notice: GROQ_API_KEY is not configured in .env"
                                ),
                                interactive=False,
                            )
                            ai_pill = gr.Textbox(
                                label="AI Status",
                                value="IDLE",
                                interactive=False,
                                visible=False,
                            )

                        with gr.Group():
                            gr.Markdown("#### Step 2: Interpreted Communication Candidates")
                            with gr.Row():
                                orig_transcript_display = gr.Textbox(
                                    label="Original Verbatim Spoken Words (Unchanged)",
                                    placeholder="Spoken words appear here...",
                                    lines=2,
                                    interactive=False,
                                    scale=1,
                                )
                                proposed_msg_box = gr.Textbox(
                                    label="Proposed Message (Editable by Patient)",
                                    placeholder="Groq's primary interpretation appears here...",
                                    lines=2,
                                    interactive=True,
                                    scale=1,
                                )

                            gr.Markdown("##### Ranked Communication Options (Tap to Select)")
                            with gr.Row():
                                cand_btn_1 = gr.Button("1. [Awaiting Input]", variant="secondary", scale=1)
                                cand_btn_2 = gr.Button("2. [Awaiting Input]", variant="secondary", scale=1)
                            with gr.Row():
                                cand_btn_3 = gr.Button("3. [Awaiting Input]", variant="secondary", scale=1)
                                cand_btn_4 = gr.Button("4. [Awaiting Input]", variant="secondary", scale=1)

                            with gr.Accordion("♿ Accessible Number & Dwell Selection", open=False):
                                with gr.Row():
                                    quick_key_1 = gr.Button("⌨️ Key [1]", variant="secondary", scale=1)
                                    quick_key_2 = gr.Button("⌨️ Key [2]", variant="secondary", scale=1)
                                    quick_key_3 = gr.Button("⌨️ Key [3]", variant="secondary", scale=1)
                                    quick_key_4 = gr.Button("⌨️ Key [4]", variant="secondary", scale=1)

                                with gr.Row():
                                    dwell_enabled_cb = gr.Checkbox(label="Enable Dwell Selection", value=False)
                                    dwell_duration_slider = gr.Slider(minimum=1.0, maximum=3.5, value=1.5, step=0.5, label="Dwell Duration (seconds)")
                                    gesture_selection_optin_cb = gr.Checkbox(
                                        label="Opt-in: Gesture Candidate Selection",
                                        value=False,
                                    )

                                with gr.Row():
                                    dwell_target_dropdown = gr.Dropdown(
                                        label="Dwell Target Item",
                                        choices=["Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4", "Confirm Message"],
                                        value="Candidate 1",
                                        scale=2
                                    )
                                    simulate_dwell_btn = gr.Button("⏳ Start Dwell Focus", variant="primary", scale=1)
                                    cancel_dwell_btn = gr.Button("🛑 Cancel Dwell", variant="stop", scale=1)

                                dwell_feedback_box = gr.Textbox(
                                    label="Dwell Telemetry",
                                    value="Accessible input ready. Standard tap, keyboard [1-4], or dwell available.",
                                    interactive=False,
                                    lines=1
                                )

                            gr.Markdown("##### Conversation Partner Context & Reply Suggestions")
                            partner_context_display = gr.Textbox(
                                label="What Partner Said",
                                placeholder="When speaker is 'Conversation Partner', their speech appears here...",
                                lines=2,
                                interactive=False,
                            )
                            with gr.Row():
                                reply_btn_yes = gr.Button("👍 Yes / Agree: [None]", variant="secondary", scale=1)
                                reply_btn_no = gr.Button("👎 No / Disagree: [None]", variant="secondary", scale=1)
                            with gr.Row():
                                reply_btn_spec = gr.Button("💬 Specific: [None]", variant="secondary", scale=1)
                                reply_btn_clar = gr.Button("❓ Clarify: [None]", variant="secondary", scale=1)

                            candidate_details_box = gr.Textbox(
                                label="Selected Candidate / Reply Evidence & Notes",
                                placeholder="Explanation of why this message was suggested...",
                                lines=2,
                                interactive=False,
                            )

                            with gr.Row():
                                detected_lang_box = gr.Textbox(label="Detected Language", value="None", interactive=False, scale=1)
                                used_mods_box = gr.Textbox(label="Modalities Used", value="None", interactive=False, scale=1)
                            with gr.Row():
                                reasoning_summary_box = gr.Textbox(label="Signal Reasoning", placeholder="Signal cues...", lines=2, interactive=False, scale=1)
                                uncertainty_box = gr.Textbox(label="Uncertainty & Ambiguities", placeholder="Ambiguity warnings...", lines=2, interactive=False, scale=1)

                        with gr.Group():
                            gr.Markdown("#### Step 3: Review & Confirmation Safeguard")
                            with gr.Row():
                                confirm_btn = gr.Button("✅ Confirm Message", variant="primary", scale=2)
                                edit_btn = gr.Button("✏️ Update Edited Text", variant="secondary", scale=1)
                                reject_btn = gr.Button("❌ Reject", variant="stop", scale=1)
                                clarify_btn = gr.Button("❓ Clarify", variant="secondary", scale=1)

                            action_status_box = gr.Textbox(
                                label="Confirmation Safeguard Status",
                                value="Awaiting interpretation or patient confirmation. Speech synthesis is locked.",
                                interactive=False,
                            )

                        with gr.Group():
                            gr.Markdown("#### Step 4: Speech Output (ElevenLabs Flash TTS)")
                            confirmed_msg_display = gr.Textbox(
                                label="Confirmed Message (Ready to Speak)",
                                value="",
                                placeholder="Confirm a message above to unlock speech synthesis...",
                                interactive=False,
                                lines=2,
                            )
                            with gr.Row():
                                speak_btn = gr.Button(
                                    "🗣️ Speak Message (Flash TTS)",
                                    variant="primary",
                                    interactive=False,
                                )

                            audio_playback = gr.Audio(
                                label="Synthesized Voice Audio",
                                type="filepath",
                                interactive=False,
                                autoplay=False,
                            )
                            tts_status_box = gr.Textbox(
                                label="TTS Status",
                                value=f"ElevenLabs Flash TTS ({flash_tts.model_id}) ready. Confirmation required.",
                                interactive=False,
                            )

                    # Right column: Live Camera Feed, Diagnostics & Conversation History
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("#### Live Vision Stream")
                            video_display = gr.Image(
                                label="Webcam View (Hand & Face Tracking)",
                                streaming=True,
                                type="numpy",
                                interactive=False,
                            )
                            with gr.Row():
                                start_cam_btn = gr.Button("🎥 Start Camera", variant="primary")
                                stop_cam_btn = gr.Button("⏹ Stop Camera", variant="secondary")

                            with gr.Row():
                                cam_status_box = gr.Textbox(label="Camera Status", value="Disconnected", interactive=False, scale=1)
                                live_gesture_box = gr.Textbox(label="Active Gesture Match", value="None", interactive=False, scale=2)

                            with gr.Accordion("🔍 Advanced Diagnostics & Sensor Telemetry", open=False):
                                hand_status_box = gr.Textbox(label="Hand Landmark Detection", value="Camera unavailable", interactive=False)
                                face_status_box = gr.Textbox(label="Face Landmark Detection", value="Camera unavailable", interactive=False)
                                head_movement_box = gr.Textbox(label="Head Movement Signals (Cues Only)", value="Camera unavailable", interactive=False)
                                lip_activity_box = gr.Textbox(label="Lip Activity & Syllable Timing (Heuristic)", value="Camera unavailable", interactive=False)

                        with gr.Group():
                            gr.Markdown("#### Session Conversation History")
                            history_box = gr.Textbox(
                                label="Confirmed Dialogue Turns",
                                value="No messages confirmed in this session yet.",
                                lines=14,
                                interactive=False,
                            )
                            clear_history_btn = gr.Button("🗑 Clear Session History", variant="secondary")


                # Timer for live camera streaming (~16 FPS)
        frame_timer = gr.Timer(value=0.06, active=False)

        # Temporary buffer for in-progress calibration samples in memory
        temp_calibration_samples = gr.State(value=[])

        # --- Helper to compute current modality summary ---
        def get_modality_summary(state: SessionState) -> str:
            lines = []
            if state.latest_raw_transcript:
                lines.append(f"• Speech: Present (\"{state.latest_raw_transcript[:40]}...\")")
            else:
                lines.append("• Speech: None recorded")

            if not state.vision_freshness.check_staleness():
                lines.append(f"• Vision Stream: Active (Hands: {state.hand_status}, Face: {state.face_status})")
            else:
                lines.append("• Vision Stream: Stopped / Stale (Visual cues unavailable)")

            if "Recognized" in state.latest_recognized_gesture:
                lines.append(f"• Calibrated Gesture: {state.latest_recognized_gesture} (Meaning: \"{state.latest_recognized_meaning}\")")
            elif "Uncertain" in state.latest_recognized_gesture:
                lines.append("• Calibrated Gesture: Uncertain / Unclassified")
            else:
                lines.append(f"• Calibrated Gesture: {state.latest_recognized_gesture}")

            return "\n".join(lines)

        # --- Camera Callbacks ---
        def handle_start_cam(state: SessionState) -> Tuple[str, str, str, str, str, str, str, gr.Timer, SessionState]:
            success, message = camera_manager.start()
            if success:
                if not gesture_detector.is_initialized:
                    gesture_detector._initialize_detector()
                if not face_detector.is_initialized:
                    face_detector._initialize_detector()
                state.vision_freshness.update(available=True, summary="Camera connected")
                state.is_vision_stale = False
                return (
                    "Connected",
                    "Connected",
                    "LIVE (Camera Active)",
                    "No hand detected",
                    "No face detected",
                    "Steady",
                    "Resting / Closed (Aperture: 0.00)",
                    gr.Timer(active=True),
                    state,
                )
            else:
                state.vision_freshness.update(available=False, summary="Camera failed to connect")
                state.is_vision_stale = True
                return (
                    "Error",
                    "Error",
                    "ERROR (Failed)",
                    "Camera unavailable",
                    "Camera unavailable",
                    "Camera unavailable",
                    "Camera unavailable",
                    gr.Timer(active=False),
                    state,
                )

        def handle_stop_cam(state: SessionState) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], str, str, str, str, str, str, str, str, gr.Timer, SessionState
        ]:
            success, message = camera_manager.release()
            cam_status = "Disconnected" if success else "Error"
            state.vision_freshness.update(available=False, summary="Camera stopped")
            state.is_vision_stale = True
            state.latest_hand_features.clear()
            state.latest_head_features.clear()
            state.face_status = "Camera unavailable"
            state.hand_status = "Camera unavailable"
            state.latest_recognized_gesture = "Camera unavailable"
            state.latest_recognized_meaning = None
            return (
                None,
                None,
                cam_status,
                cam_status,
                "OFF (Disconnected)",
                "Camera unavailable",
                "None",
                "Camera unavailable",
                "Camera unavailable",
                "Camera unavailable",
                gr.Timer(active=False),
                state,
            )

        def stream_frame(state: SessionState) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], str, str, str, str, str, str, str, str, SessionState
        ]:
            if not camera_manager.is_active:
                state.vision_freshness.update(available=False, summary="Camera idle")
                state.is_vision_stale = True
                return (
                    None,
                    None,
                    "Disconnected",
                    "Disconnected",
                    "Camera unavailable",
                    "None",
                    "Camera unavailable",
                    "Camera unavailable",
                    "Camera unavailable",
                    "NONE (Camera Stopped)",
                    state,
                )

            success, frame, message = camera_manager.read_frame()
            if not success or frame is None:
                camera_manager.release()
                state.vision_freshness.update(available=False, summary="Frame read failure")
                state.is_vision_stale = True
                return (
                    None,
                    None,
                    "Error",
                    "Error",
                    "Camera unavailable",
                    "None",
                    "Camera unavailable",
                    "Camera unavailable",
                    "Camera unavailable",
                    "ERROR (Frame Dropped)",
                    state,
                )

            hand_annotated, hand_status, hands_info = gesture_detector.process_frame(frame)
            combined_frame, face_status, movement_data = face_detector.process_frame(hand_annotated)

            state.latest_hand_features = hands_info
            state.latest_head_features = movement_data
            state.face_status = face_status
            state.hand_status = hand_status
            state.is_vision_stale = False
            state.vision_freshness.update(available=True, summary=f"Live ({hand_status}, {face_status})")

            # Real-time gesture classification using patient-calibrated dictionary
            live_feat_vec = hands_info[0].get("feature_vector") if hands_info else None
            rec_status, rec_meaning, conf, diag = gesture_classifier.classify_frame(live_feat_vec)

            state.latest_recognized_gesture = rec_status
            state.latest_recognized_meaning = rec_meaning
            state.latest_gesture_confidence = conf

            if rec_meaning:
                gesture_display_str = f"{rec_status} ➔ \"{rec_meaning}\" (Match: {conf:.2f})"
                gesture_pill_str = f"ACTIVE ({rec_status})"
                state.gesture_freshness.update(available=True, summary=f"{rec_status} ({rec_meaning})")
            elif "Uncertain" in rec_status:
                gesture_display_str = f"{rec_status} (Match: {conf:.2f})"
                gesture_pill_str = "UNCERTAIN (Unclassified)"
                state.gesture_freshness.update(available=True, summary="Uncertain")
            else:
                gesture_display_str = f"{rec_status} (Match: {conf:.2f})"
                gesture_pill_str = "IDLE (No Gesture)"
                state.gesture_freshness.update(available=False, summary=rec_status)

            lip_info = movement_data.get("lip_activity", {})
            if face_status == "Face detected":
                head_str = (
                    f"{movement_data.get('details', 'Steady')} "
                    f"[dx: {movement_data.get('delta_x', 0.0)}px, dy: {movement_data.get('delta_y', 0.0)}px]"
                )
                lip_str = (
                    f"{lip_info.get('lip_state', 'Steady')} "
                    f"(Aperture: {lip_info.get('current_aperture_ratio', 0.0):.2f}, "
                    f"Est Syllables: ~{lip_info.get('estimated_syllables_heuristic', 0)})"
                )
            else:
                head_str = "No face visible"
                lip_str = "No face visible"

            return (
                combined_frame,
                combined_frame,
                "Connected",
                "Connected",
                hand_status,
                gesture_display_str,
                face_status,
                head_str,
                lip_str,
                gesture_pill_str,
                state,
            )

        # --- Gesture Calibration Callbacks ---
        def handle_capture_sample(state: SessionState, current_samples: List[Any], name: str, meaning: str) -> Tuple[str, List[Any]]:
            if not name or not name.strip():
                return "Error: Enter a gesture name before capturing samples.", current_samples

            if not camera_manager.is_active:
                return "Error: Camera is not active. Click 'Start Camera' first to see your hand.", current_samples

            if not state.latest_hand_features:
                return "Error: No hand detected in camera. Please ensure your hand is clearly visible in the webcam.", current_samples

            feat_vec = state.latest_hand_features[0].get("feature_vector")
            if feat_vec is None:
                return "Error: Could not extract landmark feature vector from current hand frame.", current_samples

            updated_samples = list(current_samples)
            updated_samples.append(feat_vec.copy())
            sample_num = len(updated_samples)

            if sample_num > 1:
                centroid = np.mean(updated_samples, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-4:
                    centroid = centroid / norm
                sims = [float(np.dot(s, centroid)) for s in updated_samples]
                consistency = float(np.mean(sims)) * 100
                msg = f"✅ Captured sample #{sample_num} for '{name.strip()}'. Sample consistency: {consistency:.1f}%. (Capture 3-10 samples, then click 'Save Calibrated Gesture')."
            else:
                msg = f"✅ Captured sample #1 for '{name.strip()}'. Capture 2-5 more samples for higher recognition reliability."

            return msg, updated_samples

        def handle_save_gesture(
            name: str,
            meaning: str,
            captured_samples: List[Any],
            state: SessionState
        ) -> Tuple[str, str, str, str, List[Any], SessionState]:
            clean_name = name.strip() if name else ""
            clean_meaning = meaning.strip() if meaning else ""

            if not clean_name:
                return "Error: Gesture name is empty.", state.format_gestures_display(), "Error", "", captured_samples, state

            if not clean_meaning:
                return "Error: Assigned message/meaning is empty.", state.format_gestures_display(), "Error", "", captured_samples, state

            if not captured_samples or len(captured_samples) == 0:
                return f"Error: No real landmark samples captured for '{clean_name}'. Click 'Capture Current Hand Sample' first.", state.format_gestures_display(), "Error", "", captured_samples, state

            calibrated_g = gesture_classifier.add_gesture(name=clean_name, meaning=clean_meaning)
            for s in captured_samples:
                calibrated_g.add_sample(s)
                state.add_calibration_sample(gesture_name=clean_name, meaning=clean_meaning, feature_vector=s)

            consistency = calibrated_g.consistency_score() * 100
            success_msg = f"🎉 Successfully calibrated gesture '{clean_name}' with {calibrated_g.sample_count} samples! Consistency: {consistency:.1f}%."

            return success_msg, state.format_gestures_display(), f"CALIBRATED ({len(gesture_classifier.gestures)})", "", [], state

        def handle_delete_gesture(gesture_name: str, state: SessionState) -> Tuple[str, str, str, SessionState]:
            target = gesture_name.strip() if gesture_name else ""
            if not target:
                return "Enter the exact gesture name to delete.", state.format_gestures_display(), "Idle", state

            removed = gesture_classifier.remove_gesture(target)
            state.remove_calibrated_gesture(target)

            if removed:
                msg = f"🗑 Deleted calibrated gesture '{target}'."
                stat = f"CALIBRATED ({len(gesture_classifier.gestures)})" if gesture_classifier.gestures else "NONE (No Gestures)"
                return msg, state.format_gestures_display(), stat, state
            else:
                return f"Gesture '{target}' was not found in calibrated gestures.", state.format_gestures_display(), "Not Found", state

        def handle_clear_all_gestures(state: SessionState) -> Tuple[str, str, str, List[Any], SessionState]:
            gesture_classifier.clear_all()
            state.clear_all_gestures()
            return (
                "⚠️ All patient gesture calibrations cleared.",
                state.format_gestures_display(),
                "NONE (No Gestures)",
                [],
                state,
            )

        # --- Audio & Scribe Callbacks ---
        def handle_start_record(state: SessionState) -> Tuple[str, str, str, SessionState]:
            is_valid, dev_idx, dev_name, _ = AudioRecorder.get_default_input_device()
            success, msg = audio_recorder.start_recording()
            if success:
                status_text = f"● RECORDING ACTIVE... ({msg})"
                mic_pill_val = "RECORDING (Live Audio)"
            else:
                status_text = f"Recording Error:\n{msg}"
                mic_pill_val = "ERROR (Access Denied)"
            return dev_name, status_text, mic_pill_val, state

        def handle_stop_record(state: SessionState) -> Tuple[str, str, SessionState]:
            success, wav_bytes, msg = audio_recorder.stop_recording()
            if success and wav_bytes:
                state.audio_recorded_bytes = wav_bytes
                return f"Captured audio in-memory ({len(wav_bytes)} bytes). Ready to transcribe.", "READY (Captured)", state
            else:
                state.audio_recorded_bytes = None
                return f"Recording stopped: {msg}", "READY (Default)", state

        def handle_transcribe(speaker: str, state: SessionState) -> Tuple[str, str, str, str, SessionState]:
            audio_bytes = state.audio_recorded_bytes
            if audio_bytes is None or len(audio_bytes) == 0:
                return "", "Transcription Error: Please record audio first before transcribing.", "ERROR (No Audio)", "Awaiting speech...", state

            state.current_speaker = speaker
            success, transcript, status_msg = scribe_transcriber.transcribe(audio_bytes, language_code="eng")

            if success:
                clean_t = transcript.strip() if transcript else ""
                state.latest_raw_transcript = clean_t
                state.speech_freshness.update(available=True, summary=f"Verbatim: {clean_t}")

                # Invalidate previous confirmation when new speech arrives
                state.invalidate_confirmation()

                if speaker == "Patient":
                    state.latest_patient_transcript = clean_t
                else:
                    state.latest_partner_transcript = clean_t

                formatted = f"[{speaker}]: {clean_t}" if clean_t else f"[{speaker}]: (Silence / No speech recognized)"
                summary = get_modality_summary(state)
                return formatted, status_msg, "TRANSCRIBED (Fresh)", summary, state
            else:
                state.speech_freshness.update(available=False, summary="STT failed")
                return "", status_msg, "STT FAILED", "Speech transcription failed.", state

        # --- Refined Multimodal Intent Reasoning Callback ---
        def handle_interpret(
            speaker: str,
            state: SessionState,
        ) -> Tuple[
            str, str, str, str, str, str, str, str, str, Any, str, str, SessionState
        ]:
            transcript = state.latest_raw_transcript
            if not transcript or not transcript.strip():
                return (
                    "Interpretation Error: Transcript is empty. Please record and transcribe speech first.",
                    "",
                    "",
                    "",
                    "None",
                    "None",
                    "No speech transcript available.",
                    "",
                    "ERROR (No Transcript)",
                    gr.Button(interactive=False),
                    "",
                    "Awaiting inputs...",
                    state,
                )

            # Invalidate previous confirmation when generating new reasoning
            state.invalidate_confirmation()

            # Check vision staleness
            state.vision_freshness.check_staleness()

            # Extract recognized calibrated gesture if present and not stale
            calib_name = None
            calib_meaning = state.latest_recognized_meaning
            if "Recognized: " in state.latest_recognized_gesture and not state.vision_freshness.is_stale:
                calib_name = state.latest_recognized_gesture.replace("Recognized: ", "").strip()

            success, resp, msg = groq_assistant.interpret_multimodal_intent(
                transcript=transcript,
                speaker=speaker,
                hand_info=state.latest_hand_features if not state.vision_freshness.is_stale else [],
                head_movement_data=state.latest_head_features if not state.vision_freshness.is_stale else {},
                face_status=state.face_status if not state.vision_freshness.is_stale else "Camera unavailable",
                is_vision_stale=state.vision_freshness.is_stale,
                calibrated_gesture_name=calib_name,
                calibrated_gesture_meaning=calib_meaning,
                gesture_status=state.latest_recognized_gesture if not state.vision_freshness.is_stale else "Camera unavailable",
                conversation_history=state.get_history_for_prompt(),
                patient_profile=state.patient_profile.model_dump() if state.patient_profile.enabled else None,
            )

            if not success or resp is None:
                return (
                    f"Groq Status: {msg}",
                    transcript,
                    "",
                    gr.Button(value="1. [Unavailable]", interactive=False),
                    gr.Button(value="2. [Unavailable]", interactive=False),
                    gr.Button(value="3. [Unavailable]", interactive=False),
                    gr.Button(value="4. [Unavailable]", interactive=False),
                    state.latest_partner_transcript or "None",
                    gr.Button(value="👍 Yes: [None]", interactive=False),
                    gr.Button(value="👎 No: [None]", interactive=False),
                    gr.Button(value="💬 Specific: [None]", interactive=False),
                    gr.Button(value="❓ Clarify: [None]", interactive=False),
                    "Groq inference failed.",
                    "Unknown",
                    "None",
                    "Groq inference failed.",
                    msg,
                    "FAILED",
                    gr.Button(interactive=False),
                    "",
                    get_modality_summary(state),
                    state,
                )

            state.latest_interpreted_message = resp.interpreted_message
            state.alternative_interpretations = resp.alternative_interpretations
            state.detected_language = resp.language
            state.reasoning_summary = resp.reasoning_summary
            state.used_modalities = resp.used_modalities
            state.uncertainty_notes = resp.uncertainty_notes
            state.requires_clarification = resp.requires_clarification
            state.clarification_question = resp.clarification_question
            state.proposed_message = resp.interpreted_message
            state.ranked_candidates = [c.model_dump() for c in resp.candidates]
            state.selected_candidate_rank = 1

            # Format 4 candidate buttons
            cands = resp.candidates
            btn1_label = f"1. \"{cands[0].text}\" [{cands[0].intent}]" if len(cands) > 0 else "1. [None]"
            btn2_label = f"2. \"{cands[1].text}\" [{cands[1].intent}]" if len(cands) > 1 else "2. [None]"
            btn3_label = f"3. \"{cands[2].text}\" [{cands[2].intent}]" if len(cands) > 2 else "3. [None]"
            btn4_label = f"4. \"{cands[3].text}\" [{cands[3].intent}]" if len(cands) > 3 else "4. [None]"

            # Initial details for candidate 1
            cand1_details = ""
            if cands:
                cand1_details = f"Rank 1: \"{cands[0].text}\"\n• Intent: {cands[0].intent}\n• Evidence: {cands[0].why}"
                if cands[0].uncertainty_note:
                    cand1_details += f"\n• Uncertainty: {cands[0].uncertainty_note}"

            # Format uncertainty & clarification notes
            unc_parts = []
            if resp.requires_clarification and resp.clarification_question:
                unc_parts.append(f"❓ Clarification Request: {resp.clarification_question}")
            if resp.uncertainty_notes:
                unc_parts.append(f"Notice: {resp.uncertainty_notes}")
            unc_str = "\n".join(unc_parts) if unc_parts else "No significant uncertainty detected."

            # Check if speaker was Conversation Partner and generate reply suggestions if available
            partner_text = state.latest_partner_transcript if speaker == "Conversation Partner" else (state.latest_partner_transcript or "None")
            yes_lbl, no_lbl, spec_lbl, clar_lbl = "👍 Yes: [None]", "👎 No: [None]", "💬 Specific: [None]", "❓ Clarify: [None]"
            yes_inter, no_inter, spec_inter, clar_inter = False, False, False, False

            if speaker == "Conversation Partner" and transcript:
                ok_rep, rep_resp, rep_msg = groq_assistant.generate_reply_suggestions(
                    partner_transcript=transcript,
                    conversation_history=state.get_history_for_prompt(),
                    patient_profile=state.patient_profile.model_dump() if state.patient_profile.enabled else None,
                )
                if ok_rep and rep_resp:
                    state.reply_suggestions = [s.model_dump() for s in rep_resp.suggestions]
                    for s in rep_resp.suggestions:
                        cat = s.category.lower()
                        if cat == "yes":
                            yes_lbl = f"👍 Yes: \"{s.text}\""
                            yes_inter = True
                        elif cat == "no":
                            no_lbl = f"👎 No: \"{s.text}\""
                            no_inter = True
                        elif cat == "specific":
                            spec_lbl = f"💬 Specific: \"{s.text}\""
                            spec_inter = True
                        elif cat == "clarification":
                            clar_lbl = f"❓ Clarify: \"{s.text}\""
                            clar_inter = True

            mods_str = ", ".join(resp.used_modalities) if resp.used_modalities else "Speech"
            summary_str = get_modality_summary(state)

            return (
                f"Groq Status: {msg}",
                transcript,
                resp.interpreted_message,
                gr.Button(value=btn1_label, interactive=True),
                gr.Button(value=btn2_label, interactive=(len(cands) > 1)),
                gr.Button(value=btn3_label, interactive=(len(cands) > 2)),
                gr.Button(value=btn4_label, interactive=(len(cands) > 3)),
                partner_text,
                gr.Button(value=yes_lbl, interactive=yes_inter),
                gr.Button(value=no_lbl, interactive=no_inter),
                gr.Button(value=spec_lbl, interactive=spec_inter),
                gr.Button(value=clar_lbl, interactive=clar_inter),
                cand1_details,
                resp.language,
                mods_str,
                resp.reasoning_summary,
                unc_str,
                "INTERPRETED (Refined)",
                gr.Button(interactive=False),  # Strictly disabled until patient clicks Confirm
                "",  # Clear confirmed text
                summary_str,
                state,
            )

        def handle_select_candidate(candidate_idx: int, state: SessionState) -> Tuple[
            str, str, Any, str, SessionState
        ]:
            """Selects a ranked candidate, updates proposed text, and invalidates any previous confirmation."""
            state.invalidate_confirmation()
            if not state.ranked_candidates or candidate_idx >= len(state.ranked_candidates):
                return state.proposed_message, "Candidate not found.", gr.Button(interactive=False), "LOCKED", state

            cand = state.ranked_candidates[candidate_idx]
            selected_text = cand.get("text", "")
            state.proposed_message = selected_text
            state.selected_candidate_rank = cand.get("rank", candidate_idx + 1)

            details = f"Selected Rank {cand.get('rank')}: \"{selected_text}\"\n• Intent: {cand.get('intent')}\n• Evidence: {cand.get('why')}"
            if cand.get("uncertainty_note"):
                details += f"\n• Uncertainty: {cand.get('uncertainty_note')}"

            return (
                selected_text,
                details,
                gr.Button(interactive=False),
                f"LOCKED (Rank {cand.get('rank')} Selected — Confirmation required)",
                state,
            )

        # --- Dwell & Accessible Selection Callbacks (Milestone 8) ---
        def handle_execute_dwell(
            dwell_enabled: bool,
            duration: float,
            target: str,
            speaker: str,
            state: SessionState,
        ) -> Tuple[str, str, Any, str, str, SessionState]:
            """
            Executes dwell selection for a configured item after duration.
            Includes explicit cancellation check and invalidates confirmation.
            """
            if not dwell_enabled:
                return (
                    state.proposed_message,
                    "",
                    gr.Button(interactive=False),
                    "LOCKED",
                    "Notice: Dwell Selection is currently disabled. Toggle 'Enable Dwell Selection' first.",
                    state,
                )

            # Wait / dwell simulation
            time.sleep(min(duration, 3.5))

            if target.startswith("Candidate"):
                try:
                    idx = int(target.split()[-1]) - 1
                except ValueError:
                    idx = 0
                sel_text, details, btn, lock_status, updated_state = handle_select_candidate(idx, state)
                fb = f"✅ Dwell complete ({duration:.1f}s): Selected {target} -> \"{sel_text[:30]}...\". Ready for confirmation."
                return sel_text, details, btn, lock_status, fb, updated_state

            elif target == "Confirm Message":
                # Patient explicitly dwelling on Confirm action
                if not state.proposed_message or not state.proposed_message.strip():
                    return (
                        "",
                        "",
                        gr.Button(interactive=False),
                        "LOCKED",
                        "Dwell Confirm Error: No proposed message to confirm.",
                        state,
                    )
                # Confirm message
                state.confirm_message(speaker=speaker, confirmed_text=state.proposed_message)
                fb = f"✅ Dwell complete ({duration:.1f}s): Message CONFIRMED via dwell selection. TTS Unlocked."
                return (
                    state.proposed_message,
                    "Confirmed via Dwell Selection",
                    gr.Button(interactive=True),
                    "CONFIRMED (Ready to Speak)",
                    fb,
                    state,
                )

            return (
                state.proposed_message,
                "",
                gr.Button(interactive=False),
                "LOCKED",
                f"Dwell target '{target}' unhandled.",
                state,
            )

        def handle_cancel_dwell(state: SessionState) -> Tuple[str, SessionState]:
            state.invalidate_confirmation()
            return "🛑 Dwell selection cancelled/aborted. Confirmation locked.", state

        def handle_select_reply(category: str, state: SessionState) -> Tuple[
            str, str, Any, str, SessionState
        ]:
            """Selects a suggested reply to the conversation partner, populates proposed text, and invalidates confirmation."""
            state.invalidate_confirmation()
            matching = [s for s in state.reply_suggestions if s.get("category", "").lower() == category.lower()]
            if not matching:
                return state.proposed_message, f"No suggestion available for category '{category}'.", gr.Button(interactive=False), "LOCKED", state

            reply = matching[0]
            selected_text = reply.get("text", "")
            state.proposed_message = selected_text
            state.selected_reply_category = category

            details = (
                f"Selected Reply [{category.upper()}]: \"{selected_text}\"\n"
                f"• Intent: {reply.get('intent', 'other')}\n"
                f"• In Response to Partner: \"{state.latest_partner_transcript[:50]}...\""
            )

            return (
                selected_text,
                details,
                gr.Button(interactive=False),
                f"LOCKED (Reply '{category.upper()}' Selected — Confirmation required)",
                state,
            )

        # --- Patient Profile Callbacks (Milestone 3) ---
        def handle_save_profile(
            enabled: bool,
            name: str,
            lang: str,
            script: str,
            needs_str: str,
            phrases_str: str,
            notes: str,
            state: SessionState,
        ) -> Tuple[str, SessionState]:
            needs = [n.strip() for n in needs_str.split(",") if n.strip()] if needs_str else []
            phrases = [p.strip() for p in phrases_str.splitlines() if p.strip()] if phrases_str else []

            state.patient_profile.enabled = enabled
            state.patient_profile.display_name = name.strip() if name else ""
            state.patient_profile.preferred_language = lang
            state.patient_profile.preferred_script = script
            state.patient_profile.common_needs = needs
            state.patient_profile.frequently_used_phrases = phrases
            state.patient_profile.communication_notes = notes.strip() if notes else ""

            status = f"✅ Profile saved. Status: {'Enabled' if enabled else 'Disabled'}. {len(needs)} needs, {len(phrases)} phrases active."
            return status, state

        def handle_clear_profile(state: SessionState) -> Tuple[
            bool, str, str, str, str, str, str, str, SessionState
        ]:
            state.patient_profile.enabled = False
            state.patient_profile.display_name = ""
            state.patient_profile.preferred_language = "Auto"
            state.patient_profile.preferred_script = "Roman"
            state.patient_profile.common_needs.clear()
            state.patient_profile.frequently_used_phrases.clear()
            state.patient_profile.communication_notes = ""

            return (
                False,
                "",
                "Auto",
                "Roman",
                "",
                "",
                "",
                "🗑 Profile cleared and disabled for this session.",
                state,
            )

        # --- Quick Communication Panel Callbacks (Milestone 4) ---
        def handle_select_quick_phrase(phrase: str, state: SessionState) -> Tuple[
            str, str, Any, str, str, str, Any, str, SessionState
        ]:
            """Selects a quick communication phrase, fills previews on Tab 4 and Tab 7, and strictly locks TTS."""
            state.invalidate_confirmation()
            clean = phrase.strip() if phrase else ""
            state.proposed_message = clean
            details = f"Quick Phrase Selected: \"{clean}\". Gated by explicit confirmation." if clean else "Selection cleared."
            lock_status = f"LOCKED (Quick Phrase \"{clean}\" — Explicit confirmation required)" if clean else "Awaiting phrase selection."
            return (
                clean,  # proposed_msg_box (Tab 7)
                details,  # candidate_details_box (Tab 7)
                gr.Button(interactive=False),  # speak_btn (Tab 7)
                lock_status,  # action_status_box (Tab 7)
                clean,  # quick_phrase_preview (Tab 4)
                "",  # quick_confirmed_display (Tab 4)
                gr.Button(interactive=False),  # quick_speak_btn (Tab 4)
                lock_status,  # quick_action_status (Tab 4)
                state,
            )

        def handle_edit_quick_phrase(edited_phrase: str, state: SessionState) -> Tuple[
            str, str, Any, str, str, str, Any, str, SessionState
        ]:
            """Updates phrase preview after user edits text, invalidating previous confirmation."""
            state.invalidate_confirmation()
            clean = edited_phrase.strip() if edited_phrase else ""
            state.proposed_message = clean
            status = f"✏️ Phrase updated to: \"{clean}\". Confirmation required before speech." if clean else "Phrase cleared."
            return (
                clean,
                status,
                gr.Button(interactive=False),
                "LOCKED (Edited)",
                clean,
                "",
                gr.Button(interactive=False),
                status,
                state,
            )

        def handle_confirm_quick_phrase(phrase_to_confirm: str, state: SessionState) -> Tuple[
            str, str, Any, str, str, str, Any, str, SessionState
        ]:
            """Explicit confirmation for quick phrase: unlocks TTS buttons on both Tab 4 and Tab 7."""
            clean = phrase_to_confirm.strip() if phrase_to_confirm else ""
            if not clean:
                return (
                    "",
                    "Cannot confirm empty phrase.",
                    gr.Button(interactive=False),
                    "LOCKED (Empty)",
                    "",
                    "",
                    gr.Button(interactive=False),
                    "Cannot confirm empty phrase. Select a phrase first.",
                    state,
                )
            state.record_confirmed_message(speaker="Patient", confirmed_text=clean)
            status = f"✅ Confirmed: \"{clean}\". TTS unlocked."
            return (
                clean,
                status,
                gr.Button(interactive=True),  # speak_btn (Tab 7) unlocked
                "CONFIRMED ✅ (Unlocked)",
                clean,
                clean,  # quick_confirmed_display (Tab 4)
                gr.Button(interactive=True),  # quick_speak_btn (Tab 4) unlocked
                status,
                state,
            )

        def handle_cancel_quick_phrase(state: SessionState) -> Tuple[
            str, str, Any, str, str, str, Any, str, SessionState
        ]:
            """Cancels/clears current phrase selection and locks TTS."""
            state.invalidate_confirmation()
            state.proposed_message = ""
            status = "Selection cancelled and cleared. TTS locked."
            return (
                "",
                status,
                gr.Button(interactive=False),
                "LOCKED",
                "",
                "",
                gr.Button(interactive=False),
                status,
                state,
            )

        def handle_add_custom_phrase(new_phrase: str, state: SessionState) -> Tuple[
            str, Any, str, str, SessionState
        ]:
            if not new_phrase or not new_phrase.strip():
                dropdown_update = gr.Dropdown(choices=state.quick_phrases.custom, value=None)
                return "", dropdown_update, state.quick_phrases.custom, "Cannot add empty phrase.", state

            clean = new_phrase.strip()
            if clean not in state.quick_phrases.custom:
                state.quick_phrases.custom.append(clean)

            custom_list_str = "\n".join([f"• {p}" for p in state.quick_phrases.custom])
            dropdown_update = gr.Dropdown(choices=state.quick_phrases.custom, value=clean)
            return (
                "",
                dropdown_update,
                custom_list_str,
                f"✅ Added custom phrase: \"{clean}\"",
                state,
            )

        def handle_remove_custom_phrase(selected_to_remove: str, state: SessionState) -> Tuple[
            Any, str, str, SessionState
        ]:
            if not selected_to_remove or selected_to_remove not in state.quick_phrases.custom:
                dropdown_update = gr.Dropdown(choices=state.quick_phrases.custom, value=None)
                return dropdown_update, "\n".join([f"• {p}" for p in state.quick_phrases.custom]), "Please select a custom phrase to remove.", state

            state.quick_phrases.custom.remove(selected_to_remove)
            remaining = state.quick_phrases.custom
            custom_list_str = "\n".join([f"• {p}" for p in remaining]) if remaining else "No custom phrases added yet."
            dropdown_update = gr.Dropdown(choices=remaining, value=None)
            return dropdown_update, custom_list_str, f"🗑 Removed custom phrase: \"{selected_to_remove}\"", state

        # --- Voice Consent & Personalization Callbacks (Milestone 5) ---
        VOICE_ID_MAP = {
            "Roger": "CwhRBWXzGAHq8TQ4Fs17",
            "Sarah": "EXAVITQu4vr4xnSDxMaL",
            "Laura": "FGY2WhTYpPnrIDTdsKH5",
            "Charlie": "IKne3meq5aSn9XLyUdCD",
            "George": "JBFqnCBsd6RMkjVDRZzb",
            "Alice": "Xb7hH8MSUJpSbSDYk0k2",
        }

        def handle_apply_voice(selected_choice: str, state: SessionState) -> Tuple[str, SessionState]:
            voice_name = selected_choice.split(" ")[0] if selected_choice else "Roger"
            voice_id = VOICE_ID_MAP.get(voice_name, "CwhRBWXzGAHq8TQ4Fs17")
            state.voice_consent.provider_voice_id = voice_id
            state.voice_consent.voice_label = selected_choice
            state.voice_consent.voice_type = "premade"
            return f"✅ Applied active synthesis voice: {selected_choice} (ID: {voice_id})", state

        def handle_check_cloning_entitlement() -> str:
            can_clone, explanation = flash_tts.check_cloning_entitlement()
            return f"Entitlement Status: {'Eligible' if can_clone else 'Restricted'}\n{explanation}"

        def handle_submit_consent(consented: bool, signer_name: str, state: SessionState) -> Tuple[str, SessionState]:
            if not consented:
                return "❌ Consent Error: You must explicitly check the affirmative consent box before requesting voice cloning.", state

            if not signer_name or len(signer_name.strip()) < 2:
                return "❌ Identity Error: Please type your full legal name to verify and sign the consent record.", state

            can_clone, explanation = flash_tts.check_cloning_entitlement()
            if not can_clone:
                return (
                    f"⚠️ Consent Recorded for {signer_name.strip()}, but cloning cannot proceed:\n{explanation}",
                    state,
                )

            state.voice_consent.has_consented = True
            state.voice_consent.signer_name = signer_name.strip()
            state.voice_consent.acknowledgement_text = "Affirmative consent granted by speaker."
            state.voice_consent.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            return f"✅ Verified Consent Record Created for {signer_name.strip()}.", state

        def handle_delete_voice(state: SessionState) -> Tuple[str, SessionState]:
            state.voice_consent.has_consented = False
            state.voice_consent.signer_name = ""
            state.voice_consent.provider_voice_id = "CwhRBWXzGAHq8TQ4Fs17"
            state.voice_consent.voice_label = "Roger (Premade Free-Tier Voice)"
            state.voice_consent.voice_type = "premade"
            return "🗑 Custom voice association deleted and consent revoked. Reverted cleanly to default premade voice (Roger).", state

        # --- Patient Confirmation Action Callbacks ---
        def handle_confirm(proposed_text: str, speaker: str, state: SessionState) -> Tuple[
            str, str, str, Any, str, SessionState
        ]:
            if not proposed_text or not proposed_text.strip():
                return (
                    "Safety Block: Cannot confirm an empty message.",
                    state.format_log_display(),
                    "",
                    gr.Button(interactive=False),
                    "LOCKED (Empty)",
                    state,
                )

            clean_text = proposed_text.strip()
            state.record_confirmed_message(speaker=speaker, confirmed_text=clean_text)

            action_status = f"✅ Message confirmed by {speaker}: \"{clean_text}\". You may now click 'Speak Message'."
            return (
                action_status,
                state.format_log_display(),
                clean_text,
                gr.Button(interactive=True),
                "CONFIRMED ✅ (Unlocked)",
                state,
            )

        def handle_edit(edited_text: str, state: SessionState) -> Tuple[
            str, str, Any, str, SessionState
        ]:
            state.invalidate_confirmation()
            clean_text = edited_text.strip() if edited_text else ""
            state.proposed_message = clean_text

            if not clean_text:
                return (
                    "Edited text is empty. Enter message before updating.",
                    "",
                    gr.Button(interactive=False),
                    "LOCKED (Empty)",
                    state,
                )

            return (
                f"✏️ Proposed message updated to: \"{clean_text}\". Confirmation invalidated — click 'Confirm Message' to re-confirm.",
                "",
                gr.Button(interactive=False),
                "LOCKED (Re-confirmation required)",
                state,
            )

        def handle_reject(state: SessionState) -> Tuple[
            str, str, str, str, str, Any, str, SessionState
        ]:
            state.invalidate_confirmation()
            state.proposed_message = ""
            state.alternative_interpretations.clear()
            return (
                "❌ Interpretation rejected by patient. Nothing confirmed.",
                "",
                "",
                "",
                "",
                gr.Button(interactive=False),
                "REJECTED ❌ (Locked)",
                state,
            )

        def handle_clarify(clarification_text: str, state: SessionState) -> Tuple[str, SessionState]:
            state.invalidate_confirmation()
            if clarification_text and "No significant uncertainty" not in clarification_text:
                return (
                    f"❓ Clarification requested: \"{clarification_text}\". Please speak or gesture again.",
                    state,
                )
            return "❓ Patient requested clarification. Please speak or gesture again.", state

        def handle_clear_history(state: SessionState) -> Tuple[str, str, str, str, Any, str, SessionState]:
            state.clear()
            return (
                state.format_log_display(),
                "",
                "",
                "",
                gr.Button(interactive=False),
                "CLEARED (Locked)",
                state,
            )

        # --- TTS Synthesis Callback ---
        def handle_speak(state: SessionState) -> Tuple[Optional[str], str, SessionState]:
            if not state.is_confirmed or not state.confirmed_message:
                return None, "Safety Block: No confirmed message to speak. Please confirm a message first.", state

            text_to_speak = state.confirmed_message
            target_voice = state.voice_consent.provider_voice_id or None
            success, audio_bytes, msg = flash_tts.synthesize_speech(text_to_speak, voice_id=target_voice)
            if not success or not audio_bytes:
                return None, f"TTS Failed: {msg}", state

            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_file.write(audio_bytes)
            temp_file.close()

            state.latest_tts_audio_path = temp_file.name
            state.mark_spoken()

            return temp_file.name, f"✅ Speech synthesized successfully. {msg}", state

        # --- Event Bindings ---
        # Camera in Tab 1
        start_cam_btn.click(
            fn=handle_start_cam,
            inputs=[session_state],
            outputs=[
                cam_status_box,
                calib_cam_status,
                cam_pill,
                hand_status_box,
                face_status_box,
                head_movement_box,
                lip_activity_box,
                frame_timer,
                session_state,
            ],
        )

        stop_cam_btn.click(
            fn=handle_stop_cam,
            inputs=[session_state],
            outputs=[
                video_display,
                calib_video_display,
                cam_status_box,
                calib_cam_status,
                cam_pill,
                hand_status_box,
                live_gesture_box,
                face_status_box,
                head_movement_box,
                lip_activity_box,
                frame_timer,
                session_state,
            ],
        )

        # Camera in Tab 2 (Patient Gesture Calibration)
        calib_start_cam_btn.click(
            fn=handle_start_cam,
            inputs=[session_state],
            outputs=[
                cam_status_box,
                calib_cam_status,
                cam_pill,
                hand_status_box,
                face_status_box,
                head_movement_box,
                lip_activity_box,
                frame_timer,
                session_state,
            ],
        )

        calib_stop_cam_btn.click(
            fn=handle_stop_cam,
            inputs=[session_state],
            outputs=[
                video_display,
                calib_video_display,
                cam_status_box,
                calib_cam_status,
                cam_pill,
                hand_status_box,
                live_gesture_box,
                face_status_box,
                head_movement_box,
                lip_activity_box,
                frame_timer,
                session_state,
            ],
        )

        # Frame Timer streams to both displays
        frame_timer.tick(
            fn=stream_frame,
            inputs=[session_state],
            outputs=[
                video_display,
                calib_video_display,
                cam_status_box,
                calib_cam_status,
                hand_status_box,
                live_gesture_box,
                face_status_box,
                head_movement_box,
                lip_activity_box,
                gesture_pill,
                session_state,
            ],
        )

        # Gesture Calibration
        capture_sample_btn.click(
            fn=handle_capture_sample,
            inputs=[session_state, temp_calibration_samples, gesture_name_input, gesture_meaning_input],
            outputs=[calib_feedback_box, temp_calibration_samples],
        )

        save_gesture_btn.click(
            fn=handle_save_gesture,
            inputs=[gesture_name_input, gesture_meaning_input, temp_calibration_samples, session_state],
            outputs=[
                calib_feedback_box,
                gestures_list_box,
                gesture_pill,
                gesture_name_input,
                temp_calibration_samples,
                session_state,
            ],
        )

        delete_gesture_btn.click(
            fn=handle_delete_gesture,
            inputs=[delete_gesture_name, session_state],
            outputs=[gesture_manage_status, gestures_list_box, gesture_pill, session_state],
        )

        clear_all_gestures_btn.click(
            fn=handle_clear_all_gestures,
            inputs=[session_state],
            outputs=[
                gesture_manage_status,
                gestures_list_box,
                gesture_pill,
                temp_calibration_samples,
                session_state,
            ],
        )

        # Patient Profile Controls (Milestone 3)
        profile_save_btn.click(
            fn=handle_save_profile,
            inputs=[
                profile_enable_cb,
                profile_name_input,
                profile_lang_dropdown,
                profile_script_dropdown,
                profile_needs_input,
                profile_phrases_input,
                profile_notes_input,
                session_state,
            ],
            outputs=[profile_status_box, session_state],
        )

        profile_clear_btn.click(
            fn=handle_clear_profile,
            inputs=[session_state],
            outputs=[
                profile_enable_cb,
                profile_name_input,
                profile_lang_dropdown,
                profile_script_dropdown,
                profile_needs_input,
                profile_phrases_input,
                profile_notes_input,
                profile_status_box,
                session_state,
            ],
        )

        # Quick Communication Panel Controls (Milestone 4)
        quick_outputs = [
            proposed_msg_box,
            candidate_details_box,
            speak_btn,
            action_status_box,
            quick_phrase_preview,
            quick_confirmed_display,
            quick_speak_btn,
            quick_action_status,
            session_state,
        ]

        quick_btn_yes.click(fn=lambda s: handle_select_quick_phrase("Yes.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_no.click(fn=lambda s: handle_select_quick_phrase("No.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_wait.click(fn=lambda s: handle_select_quick_phrase("Please wait.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_rep.click(fn=lambda s: handle_select_quick_phrase("Please repeat.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_thx.click(fn=lambda s: handle_select_quick_phrase("Thank you.", s), inputs=[session_state], outputs=quick_outputs)

        quick_btn_water.click(fn=lambda s: handle_select_quick_phrase("I need water.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_food.click(fn=lambda s: handle_select_quick_phrase("I am hungry.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_med.click(fn=lambda s: handle_select_quick_phrase("I need my medicine.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_wash.click(fn=lambda s: handle_select_quick_phrase("I need to use the washroom.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_tired.click(fn=lambda s: handle_select_quick_phrase("I am tired.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_hot.click(fn=lambda s: handle_select_quick_phrase("I feel hot.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_cold.click(fn=lambda s: handle_select_quick_phrase("I feel cold.", s), inputs=[session_state], outputs=quick_outputs)

        quick_btn_help.click(fn=lambda s: handle_select_quick_phrase("🚨 I need help.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_pain.click(fn=lambda s: handle_select_quick_phrase("I am in pain.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_call.click(fn=lambda s: handle_select_quick_phrase("Please call someone.", s), inputs=[session_state], outputs=quick_outputs)
        quick_btn_stop.click(fn=lambda s: handle_select_quick_phrase("Please stop.", s), inputs=[session_state], outputs=quick_outputs)

        # Tab 4 Preview, Edit, Confirm, and Cancel Handlers
        quick_confirm_btn.click(
            fn=handle_confirm_quick_phrase,
            inputs=[quick_phrase_preview, session_state],
            outputs=quick_outputs,
        )
        quick_edit_btn.click(
            fn=handle_edit_quick_phrase,
            inputs=[quick_phrase_preview, session_state],
            outputs=quick_outputs,
        )
        quick_cancel_btn.click(
            fn=handle_cancel_quick_phrase,
            inputs=[session_state],
            outputs=quick_outputs,
        )

        # Tab 4 TTS Speech playback
        quick_speak_btn.click(
            fn=handle_speak,
            inputs=[session_state],
            outputs=[quick_audio_playback, quick_tts_status, session_state],
        )

        # Custom Phrase Handlers
        add_custom_btn.click(
            fn=handle_add_custom_phrase,
            inputs=[custom_phrase_input, session_state],
            outputs=[custom_phrase_input, custom_phrase_selector, custom_phrases_display, quick_action_status, session_state],
        )
        use_custom_btn.click(
            fn=lambda phr, s: handle_select_quick_phrase(phr, s),
            inputs=[custom_phrase_selector, session_state],
            outputs=quick_outputs,
        )
        remove_custom_btn.click(
            fn=handle_remove_custom_phrase,
            inputs=[custom_phrase_selector, session_state],
            outputs=[custom_phrase_selector, custom_phrases_display, quick_action_status, session_state],
        )

        # Urgent Caregiver Help Actions (Milestone 9)
        urg_btn_help.click(
            fn=lambda s: handle_select_quick_phrase("🚨 I need urgent help!", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        urg_btn_pain.click(
            fn=lambda s: handle_select_quick_phrase("⚠️ Severe pain, please assist me.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        urg_btn_caregiver.click(
            fn=lambda s: handle_select_quick_phrase("📞 Call my caregiver immediately.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        urg_btn_breathe.click(
            fn=lambda s: handle_select_quick_phrase("🫁 Trouble breathing / dizzy.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        urg_btn_cancel.click(
            fn=handle_cancel_quick_phrase,
            inputs=[session_state],
            outputs=quick_outputs,
        )

        # Voice Personalization & Consent Controls (Milestone 5)
        apply_voice_btn.click(
            fn=handle_apply_voice,
            inputs=[active_voice_dropdown, session_state],
            outputs=[voice_selection_status, session_state],
        )

        check_cloning_btn.click(
            fn=handle_check_cloning_entitlement,
            inputs=[],
            outputs=[consent_feedback_box],
        )

        submit_consent_btn.click(
            fn=handle_submit_consent,
            inputs=[consent_checkbox, consent_signer_input, session_state],
            outputs=[consent_feedback_box, session_state],
        )

        delete_voice_btn.click(
            fn=handle_delete_voice,
            inputs=[session_state],
            outputs=[consent_feedback_box, session_state],
        )

        # Audio & Scribe
        record_btn.click(
            fn=handle_start_record,
            inputs=[session_state],
            outputs=[default_mic_box, rec_status_box, mic_pill, session_state],
        )

        stop_rec_btn.click(
            fn=handle_stop_record,
            inputs=[session_state],
            outputs=[rec_status_box, mic_pill, session_state],
        )

        transcribe_btn.click(
            fn=handle_transcribe,
            inputs=[speaker_selector, session_state],
            outputs=[raw_transcript_box, stt_status_box, stt_pill, observed_modalities_box, session_state],
        )

        # Groq Multimodal Reasoning & Retry
        interpret_btn.click(
            fn=handle_interpret,
            inputs=[speaker_selector, session_state],
            outputs=[
                ai_status_box,
                orig_transcript_display,
                proposed_msg_box,
                cand_btn_1,
                cand_btn_2,
                cand_btn_3,
                cand_btn_4,
                partner_context_display,
                reply_btn_yes,
                reply_btn_no,
                reply_btn_spec,
                reply_btn_clar,
                candidate_details_box,
                detected_lang_box,
                used_mods_box,
                reasoning_summary_box,
                uncertainty_box,
                ai_pill,
                speak_btn,
                confirmed_msg_display,
                observed_modalities_box,
                session_state,
            ],
        )

        retry_btn.click(
            fn=handle_interpret,
            inputs=[speaker_selector, session_state],
            outputs=[
                ai_status_box,
                orig_transcript_display,
                proposed_msg_box,
                cand_btn_1,
                cand_btn_2,
                cand_btn_3,
                cand_btn_4,
                partner_context_display,
                reply_btn_yes,
                reply_btn_no,
                reply_btn_spec,
                reply_btn_clar,
                candidate_details_box,
                detected_lang_box,
                used_mods_box,
                reasoning_summary_box,
                uncertainty_box,
                ai_pill,
                speak_btn,
                confirmed_msg_display,
                observed_modalities_box,
                session_state,
            ],
        )

        # Candidate selection handlers (Invalidates confirmation immediately on selection)
        cand_btn_1.click(
            fn=lambda state: handle_select_candidate(0, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        cand_btn_2.click(
            fn=lambda state: handle_select_candidate(1, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        cand_btn_3.click(
            fn=lambda state: handle_select_candidate(2, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        cand_btn_4.click(
            fn=lambda state: handle_select_candidate(3, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )

        # Accessible Keyboard Shortcut Buttons [1-4]
        quick_key_1.click(
            fn=lambda state: handle_select_candidate(0, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        quick_key_2.click(
            fn=lambda state: handle_select_candidate(1, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        quick_key_3.click(
            fn=lambda state: handle_select_candidate(2, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        quick_key_4.click(
            fn=lambda state: handle_select_candidate(3, state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )

        # Accessible Dwell Selection handlers
        simulate_dwell_btn.click(
            fn=handle_execute_dwell,
            inputs=[dwell_enabled_cb, dwell_duration_slider, dwell_target_dropdown, speaker_selector, session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, dwell_feedback_box, session_state],
        )
        cancel_dwell_btn.click(
            fn=handle_cancel_dwell,
            inputs=[session_state],
            outputs=[dwell_feedback_box, session_state],
        )

        # Other Person Reply Suggestion selection handlers (Invalidates confirmation immediately)
        reply_btn_yes.click(
            fn=lambda state: handle_select_reply("yes", state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        reply_btn_no.click(
            fn=lambda state: handle_select_reply("no", state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        reply_btn_spec.click(
            fn=lambda state: handle_select_reply("specific", state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )
        reply_btn_clar.click(
            fn=lambda state: handle_select_reply("clarification", state),
            inputs=[session_state],
            outputs=[proposed_msg_box, candidate_details_box, speak_btn, action_status_box, session_state],
        )

        # Patient Confirmation Controls
        confirm_btn.click(
            fn=handle_confirm,
            inputs=[proposed_msg_box, speaker_selector, session_state],
            outputs=[
                action_status_box,
                history_box,
                confirmed_msg_display,
                speak_btn,
                confirm_pill,
                session_state,
            ],
        )

        edit_btn.click(
            fn=handle_edit,
            inputs=[proposed_msg_box, session_state],
            outputs=[
                action_status_box,
                confirmed_msg_display,
                speak_btn,
                confirm_pill,
                session_state,
            ],
        )

        reject_btn.click(
            fn=handle_reject,
            inputs=[session_state],
            outputs=[
                action_status_box,
                proposed_msg_box,
                candidate_details_box,
                uncertainty_box,
                confirmed_msg_display,
                speak_btn,
                confirm_pill,
                session_state,
            ],
        )

        clarify_btn.click(
            fn=handle_clarify,
            inputs=[uncertainty_box, session_state],
            outputs=[action_status_box, session_state],
        )

        clear_history_btn.click(
            fn=handle_clear_history,
            inputs=[session_state],
            outputs=[
                history_box,
                orig_transcript_display,
                proposed_msg_box,
                confirmed_msg_display,
                speak_btn,
                confirm_pill,
                session_state,
            ],
        )

        # TTS Synthesis Execution
        speak_btn.click(
            fn=handle_speak,
            inputs=[session_state],
            outputs=[audio_playback, tts_status_box, session_state],
        )

        # Home Page Navigation Routes & Shortcuts
        go_quick_btn.click(fn=lambda: gr.Tabs(selected="tab_quick"), inputs=[], outputs=[main_nav_tabs])
        go_ai_btn.click(fn=lambda: gr.Tabs(selected="tab_ai_understanding"), inputs=[], outputs=[main_nav_tabs])

        # Home Shortcuts
        home_quick_help.click(
            fn=lambda s: handle_select_quick_phrase("🚨 I need help.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        home_quick_water.click(
            fn=lambda s: handle_select_quick_phrase("I need water.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        home_quick_yes.click(
            fn=lambda s: handle_select_quick_phrase("Yes.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        home_quick_no.click(
            fn=lambda s: handle_select_quick_phrase("No.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )
        home_quick_wait.click(
            fn=lambda s: handle_select_quick_phrase("Please wait.", s),
            inputs=[session_state],
            outputs=quick_outputs,
        )

    return demo
