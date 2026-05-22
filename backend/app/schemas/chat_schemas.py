from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

class Attachment(BaseModel):
    """Represents a file attachment in a chat message."""
    id: Optional[str] = Field(None, alias="_id") # Unique ID for the attachment
    filename: str
    bucket_name: str
    object_name: str
    size: int
    content_type: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    id: Optional[str] = Field(None, alias="_id") # Use _id for MongoDB compatibility
    conversation_id: Optional[str] = None # Optional as it might not be present before saving
    sender: str # 'user' or 'bot'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attachments: List[Attachment] = [] # Add attachments field
    search_results: Optional[Dict[str, Any]] = None # Source and online search payload for the UI

    class Config:
        from_attributes = True
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class Conversation(BaseModel):
    """Represents a chat conversation."""
    id: Optional[str] = Field(None, alias="_id") # Use _id for MongoDB compatibility
    user_id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[ChatMessage] = []

    class Config:
        from_attributes = True
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class ConversationListResponse(BaseModel):
    """Represents a list of conversations."""
    conversations: List[Conversation]

class MessageListResponse(BaseModel):
    """Represents a list of messages in a conversation."""
    messages: List[ChatMessage]

class ConversationCreate(BaseModel):
    """Request body for creating a new conversation."""
    title: str

class MessageCreate(BaseModel):
    """Request body for creating a new message."""
    # conversation_id is now optional in MessageCreate as it will be part of the URL
    content: str
    attachments: List[Attachment] = [] # Include attachments in the message creation request
    search_ai_active: bool = False
    search_rosti_active: bool = False
    search_online_active: bool = False
    show_think_process: Optional[bool] = False # Add this line

class ConversationTitleUpdate(BaseModel):
    """Request body for updating a conversation's title."""
    title: str
