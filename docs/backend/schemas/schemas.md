# `schemas/schemas.py` - Kommentar

Hinweis`backend/app/schemas/schemas.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`pydantic.BaseModel` Hinweis

Hinweis
*   **`UserBase`**: BenutzerKommentar`username`, `email`). 
*   **`UserCreate`**: BenutzerKommentar
*   **`UserUpdate`**: BenutzerKommentar
*   **`UserInDB`**: Kommentar
*   **`Token`**: JWT Kommentar`access_token`, `token_type`). 
*   **`TokenData`**: JWT Kommentar`username`). 
*   **`Message`**: Kommentar`message`). 
*   **`FileGistBase`**: Kommentar
*   **`FileGistCreate`**: Kommentar
*   **`FileGistResponse`**: Kommentar`download_url`. 
*   **`AgentChatRequest`**: Kommentar
*   **`AgentChatResponse`**: Kommentar

```python
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from datetime import datetime

# User Schemas
class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(UserBase):
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class UserInDBBase(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True # or orm_mode = True for Pydantic v1

class User(UserInDBBase):
    pass

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

# Message Schema
class Message(BaseModel):
    message: str

# FileGist Schemas
class FileGistBase(BaseModel):
    filename: str
    file_path: str

class FileGistCreate(FileGistBase):
    pass

class FileGistResponse(FileGistBase):
    id: int
    upload_time: datetime
    user_id: int
    download_url: Optional[str] = None # Added for RAG links

    class Config:
        from_attributes = True

# Chat Schemas (from chat_schemas.py, but often combined or referenced)
class AgentChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class AgentChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    timestamp: datetime
    is_user: bool
    source_documents: Optional[List[dict]] = None # For RAG sources
```

## Kommentar
`/backend/app/schemas/schemas.py`