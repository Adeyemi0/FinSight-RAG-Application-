"""Conversation memory management for multi-turn interactions."""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class ConversationMessage:
    """Single message in a conversation."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationHistory:
    """Manages conversation history with context window management."""
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize conversation history.
        
        Args:
            max_tokens: Maximum tokens to keep in history (rough estimate)
        """
        self.messages: List[ConversationMessage] = []
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        message = ConversationMessage(role, content)
        self.messages.append(message)
        
        # Trim history if needed
        self._trim_history()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get messages in LangChain format.
        
        Returns:
            List of message dictionaries with role and content
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def get_context_summary(self) -> str:
        """
        Get a text summary of conversation context.
        
        Returns:
            Formatted conversation history
        """
        if not self.messages:
            return ""
        
        context_parts = []
        for msg in self.messages[-6:]:  # Last 3 exchanges (6 messages)
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content[:200]}...")
        
        return "\n\n".join(context_parts)
    
    def _trim_history(self):
        """
        Trim history to stay within token limits.
        
        Uses a simple heuristic: ~4 chars per token
        Keeps system messages and recent conversation
        """
        if not self.messages:
            return
        
        # Estimate total tokens (rough: 4 chars per token)
        total_chars = sum(len(msg.content) for msg in self.messages)
        estimated_tokens = total_chars / 4
        
        # If under limit, keep all
        if estimated_tokens <= self.max_tokens:
            return
        
        # Keep most recent messages that fit within limit
        chars_limit = self.max_tokens * 4
        cumulative_chars = 0
        keep_from_index = len(self.messages)
        
        # Work backwards from most recent
        for i in range(len(self.messages) - 1, -1, -1):
            msg_chars = len(self.messages[i].content)
            if cumulative_chars + msg_chars > chars_limit:
                keep_from_index = i + 1
                break
            cumulative_chars += msg_chars
        
        # Always keep at least the last 2 messages (1 exchange)
        keep_from_index = min(keep_from_index, len(self.messages) - 2)
        
        # Trim older messages
        if keep_from_index > 0:
            self.messages = self.messages[keep_from_index:]
    
    def clear(self):
        """Clear all conversation history."""
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export conversation to dictionary."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "max_tokens": self.max_tokens
        }


class SessionManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationHistory] = {}
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session ID, generates one if not provided
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = ConversationHistory()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationHistory]:
        """
        Get a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationHistory or None if not found
        """
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """
        Delete a conversation session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def clear_all_sessions(self):
        """Clear all sessions."""
        self.sessions.clear()


# Global session manager instance
session_manager = SessionManager()