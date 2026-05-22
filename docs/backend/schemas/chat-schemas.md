# `schemas/chat_schemas.py` - Kommentar

Hinweis`backend/app/schemas/chat_schemas.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **RAG Kommentar**: Kommentar

## Kommentar
Hinweis`pydantic.BaseModel` Hinweis

Hinweis
*   **`AgentChatRequest`**:
    ```python
    from typing import Optional, List
    from pydantic import BaseModel, Field
    from datetime import datetime

    class AgentChatRequest(BaseModel):
        message: str = Field(..., description="The user's message to the chat agent.")
        conversation_id: Optional[str] = Field(None, description="Optional ID of the ongoing conversation.")
    ```
*   **`SourceDocument`**:
    ```python
    class SourceDocument(BaseModel):
        filename: str = Field(..., description="The name of the source file.")
        download_url: Optional[str] = Field(None, description="Pre-signed URL to download the source file.")
        summary: Optional[str] = Field(None, description="A summary of the relevant content from the source document.")
        chunk_content: Optional[str] = Field(None, description="The specific chunk content from the source document.")
    ```
*   **`AgentChatResponse`**:
    ```python
    class AgentChatResponse(BaseModel):
        response: str = Field(..., description="The agent's response message.")
        conversation_id: str = Field(..., description="The ID of the conversation.")
        message_id: str = Field(..., description="The unique ID of the generated message.")
        timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the message.")
        is_user: bool = Field(False, description="True if the message is from the user, False if from the agent.")
        source_documents: Optional[List[SourceDocument]] = Field(None, description="List of source documents used for RAG.")
    ```

## Kommentar
`/backend/app/schemas/chat_schemas.py`