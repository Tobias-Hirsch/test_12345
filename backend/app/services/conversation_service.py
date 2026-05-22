import redis
import json
from typing import Dict, Any, Optional
from fastapi import Depends
from datetime import datetime, timedelta

from app.core.redis_client import get_redis_client
from app.core.config import settings
from app.utils.chat_message_sanitizer import sanitize_chat_history

class ConversationService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.conversation_ttl_seconds = settings.REMOVE_OLD_CONVERSATIONS_AFTER_DAYS * 24 * 60 * 60 # Convert days to seconds

    def _get_conversation_key(self, user_id: int, conversation_id: str) -> str:
        return f"conversation:{user_id}:{conversation_id}"

    def _json_serial(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError ("Type %s not serializable" % type(obj))

    def save_conversation_state(self, user_id: int, conversation_id: str, state_data: Dict[str, Any]):
        """
        Saves the conversation state to Redis.
        state_data should include:
        - "history": List of chat messages
        - "context": Current conversation context (e.g., topic, entities)
        - "variables": Session-related temporary variables
        - "attached_files": List of attached file info (e.g., file IDs, MinIO paths)
        """
        key = self._get_conversation_key(user_id, conversation_id)
        state_to_save = dict(state_data)
        state_to_save["history"] = sanitize_chat_history(state_to_save.get("history", []))
        self.redis.setex(key, self.conversation_ttl_seconds, json.dumps(state_to_save, default=self._json_serial))

    def get_conversation_state(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the conversation state from Redis.
        """
        key = self._get_conversation_key(user_id, conversation_id)
        data = self.redis.get(key)
        if data:
            state = json.loads(data)
            sanitized_history = sanitize_chat_history(state.get("history", []))
            if sanitized_history != state.get("history", []):
                state["history"] = sanitized_history
                self.save_conversation_state(user_id, conversation_id, state)
            else:
                state["history"] = sanitized_history
            return state
        return None

    def delete_conversation_state(self, user_id: int, conversation_id: str):
        """
        Deletes the conversation state from Redis.
        """
        key = self._get_conversation_key(user_id, conversation_id)
        self.redis.delete(key)

def get_conversation_service(redis_client: redis.Redis = Depends(get_redis_client)):
    """
    FastAPI dependency to get a ConversationService instance.
    """
    return ConversationService(redis_client)
