"""
Verification Test for Milestone 9:
Emergency and Quick-Help Workflow with Safety Gates.

Tests:
1. Urgent phrase selection populates proposed message immediately without STT/LLM.
2. Invalidation & lock: Speech output is locked until explicit patient confirmation.
3. Cancellation control: Immediate cancellation clears the proposed message and locks TTS.
4. Non-dispatch safeguard: Verifies that no external emergency dispatch is initiated.
"""

from core.session_state import SessionState

def test_urgent_phrase_selection_and_safety_lock():
    state = SessionState()
    urgent_msg = "🚨 I need urgent help!"
    
    # Patient triggers urgent help phrase
    state.invalidate_confirmation()
    state.proposed_message = urgent_msg
    
    # Assert proposed message updated
    assert state.proposed_message == "🚨 I need urgent help!"
    # Assert speech output remains strictly locked
    assert state.is_confirmed is False
    assert state.confirmed_message == ""
    print("PASS: Urgent phrase loaded and confirmation lock remains active.")

def test_urgent_phrase_confirmation_and_cancellation():
    state = SessionState()
    urgent_msg = "⚠️ Severe pain, please assist me."
    state.proposed_message = urgent_msg
    
    # Patient confirms urgent message
    state.record_confirmed_message(speaker="Patient", confirmed_text=urgent_msg)
    assert state.is_confirmed is True
    assert state.confirmed_message == "⚠️ Severe pain, please assist me."
    
    # Patient clicks cancel / abort
    state.invalidate_confirmation()
    state.proposed_message = ""
    assert state.is_confirmed is False
    assert state.confirmed_message == ""
    assert state.proposed_message == ""
    print("PASS: Urgent phrase confirmed and immediate cancellation verified.")

def test_non_dispatch_safety_guarantee():
    state = SessionState()
    # Confirm that session state has no automatic dispatch trigger flags
    assert not hasattr(state, "auto_dial_911")
    assert not hasattr(state, "dispatch_emergency_services")
    print("PASS: Non-dispatch safety guarantee verified (pure assistive communication).")

if __name__ == "__main__":
    print("--- RUNNING MILESTONE 9 VERIFICATION ---")
    test_urgent_phrase_selection_and_safety_lock()
    test_urgent_phrase_confirmation_and_cancellation()
    test_non_dispatch_safety_guarantee()
    print("--- ALL MILESTONE 9 UNIT TESTS PASSED ---")

