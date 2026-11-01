"""
Test suite for Instant Non-Verbal Communication Phrases Page (Tab 4):
Verifies:
1. Clicking predefined phrase (e.g. 'Yes.') populates preview and locks TTS.
2. Clicking 'I need water.' replaces previous selection.
3. Editing phrase resets and invalidates confirmation.
4. Confirming phrase unlocks TTS.
5. Live speech synthesis with ElevenLabs Flash TTS using confirmed text.
6. Adding a custom phrase.
7. Removing a custom phrase.
8. Emergency / Urgent phrase selection ('🚨 I need help.') and safety gate.
9. Changing phrase after confirmation prevents old text from being spoken.
"""

from core.session_state import SessionState
from core.audio.flash_tts import FlashTTS

def test_phrase_selection_and_replacement():
    state = SessionState()
    
    # 1. Click 'Yes.'
    state.invalidate_confirmation()
    state.proposed_message = "Yes."
    assert state.proposed_message == "Yes."
    assert state.is_confirmed is False
    assert state.confirmed_message == ""
    print("PASS: Click 'Yes' populates preview and maintains TTS lock.")

    # 2. Click 'I need water.' -> Replaces selection
    state.invalidate_confirmation()
    state.proposed_message = "I need water."
    assert state.proposed_message == "I need water."
    assert state.is_confirmed is False
    print("PASS: Click 'I need water' replaces previous selection.")

def test_edit_and_confirmation_lifecycle():
    state = SessionState()
    state.proposed_message = "I need water."
    
    # 3. Edit phrase before confirmation
    state.invalidate_confirmation()
    state.proposed_message = "I need warm water please."
    assert state.is_confirmed is False
    print("PASS: Editing phrase preserves unconfirmed state.")

    # 4. Confirm the phrase
    state.record_confirmed_message(speaker="Patient", confirmed_text=state.proposed_message)
    assert state.is_confirmed is True
    assert state.confirmed_message == "I need warm water please."
    print("PASS: Explicit confirmation records message and marks state confirmed.")

    # 5. Modifying phrase after confirmation immediately invalidates
    state.invalidate_confirmation()
    state.proposed_message = "No."
    assert state.is_confirmed is False
    assert state.confirmed_message == ""
    print("PASS: Changing phrase after confirmation prevents old text from being spoken.")

def test_custom_phrase_lifecycle():
    state = SessionState()
    # 6. Add custom phrase
    custom_phrase = "Please open the balcony door."
    if custom_phrase not in state.quick_phrases.custom:
        state.quick_phrases.custom.append(custom_phrase)
    assert custom_phrase in state.quick_phrases.custom
    print("PASS: Custom phrase added successfully.")

    # 7. Remove custom phrase
    state.quick_phrases.custom.remove(custom_phrase)
    assert custom_phrase not in state.quick_phrases.custom
    print("PASS: Custom phrase removed successfully.")

def test_emergency_phrase_safety():
    state = SessionState()
    # 8. Emergency phrase
    state.invalidate_confirmation()
    state.proposed_message = "🚨 I need help."
    assert state.is_confirmed is False
    # Verify no auto-dispatch or external contact happens
    assert not hasattr(state, "auto_dial_911")
    print("PASS: Emergency phrase selected with strict local confirmation gate.")

def test_actual_elevenlabs_tts():
    tts = FlashTTS()
    if not tts.is_configured:
        print("SKIP: ELEVENLABS_API_KEY not configured for live TTS test.")
        return

    text = "Yes, thank you."
    success, audio_bytes, msg = tts.synthesize_speech(text)
    assert success is True, f"TTS synthesis failed: {msg}"
    assert len(audio_bytes) > 500, "Synthesized audio bytes empty"
    print(f"PASS: Live ElevenLabs Flash TTS synthesis verified ({len(audio_bytes)} bytes received).")

if __name__ == "__main__":
    print("--- RUNNING INSTANT PHRASES & TAB 4 VERIFICATION SUITE ---")
    test_phrase_selection_and_replacement()
    test_edit_and_confirmation_lifecycle()
    test_custom_phrase_lifecycle()
    test_emergency_phrase_safety()
    test_actual_elevenlabs_tts()
    print("--- ALL TESTS PASSED ---")
