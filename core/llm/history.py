"""
Session Conversation History Manager.

Maintains limited, in-memory session dialogue history between Patient and
Conversation Partner without permanent disk persistence or mock conversations.
"""

from typing import List, Dict


class ConversationHistory:
    """Manages ephemeral in-memory conversation history for the current session."""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []

    def add_entry(self, speaker: str, text: str):
        """Append a real confirmed message or transcription to history."""
        clean_text = text.strip()
        if not clean_text:
            return
        self._history.append({"speaker": speaker, "text": clean_text})
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current list of history entries."""
        return list(self._history)

    def clear(self):
        """Clear all conversation history in the session."""
        self._history.clear()

    def format_display(self) -> str:
        """Format history for UI display."""
        if not self._history:
            return "No messages confirmed in this session yet."
        lines = []
        for idx, item in enumerate(self._history, start=1):
            lines.append(f"{idx}. [{item['speaker']}]: {item['text']}")
        return "\n".join(lines)

