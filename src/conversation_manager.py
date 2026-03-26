from collections import deque
from typing import Dict, List

class ConversationManager:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.sessions: Dict[str, deque] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Returns the conversation history for a given session."""
        if session_id not in self.sessions:
            return []
        return list(self.sessions[session_id])

    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message to the session's history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_history * 2) # *2 for Q&A pairs
            
        self.sessions[session_id].append({"role": role, "content": content})

    def clear_session(self, session_id: str):
        """Clears the history for a given session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
