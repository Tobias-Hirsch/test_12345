from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime # Import datetime
from fastapi import UploadFile # Import UploadFile
from enum import Enum

class Action(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MANAGE = "manage"
    EXECUTE = "execute"
    VIEW = "view"
    ALL = "all"

class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None # ErlaubenHinweis
    security_level: Optional[int] = None # Hinweis
    first_name: Optional[str] = None # Optional
    surname: Optional[str] = None # Optional

class UserCreate(UserBase):
    password: Optional[str] = None # ErlaubenHinweis
    is_active: Optional[bool] = True # Hinweis

class User(UserBase):
    id: int
    is_active: bool # Hinweis
    # department: str # Kommentar
    # first_name: Optional[str] = None # Kommentar
    # surname: Optional[str] = None # Kommentar
    avatar: Optional[str] = None # Add avatar field
    activation_token: Optional[str] = None # Add activation_token
    activation_expires_at: Optional[datetime] = None # Add activation_expires_at
    roles: list['Role'] = [] # Add roles
    provider: Optional[str] = None # Hinweis
    provider_id: Optional[str] = None # Hinweis

    class Config:
        from_attributes = True # Enable ORM mode (Pydantic V2)

class UserUpdate(BaseModel): # Inherit from BaseModel, not UserBase, for optional fields
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    department: Optional[str] = None # Allow updating department
    first_name: Optional[str] = None # Allow updating first_name
    surname: Optional[str] = None # Allow updating surname
    is_active: Optional[bool] = None # Hinweis
    avatar: Optional[str] = None # Allow updating avatar
    security_level: Optional[int] = None # Hinweis
    password: Optional[str] = None # ErlaubenHinweis

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User # Include the User schema in the Token schema

class TokenData(BaseModel):
    username: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email_or_phone: str

class PasswordResetRequest(BaseModel):
    token: str
    new_password: str

class UserLogin(BaseModel):
    username: str
    password: str

class SmtpConfig(BaseModel):
    smtp_server: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_sender_email: EmailStr # Use EmailStr for validation

class TestEmailRequest(BaseModel):
    recipient_email: EmailStr

class CaptchaVerifyRequest(BaseModel):
    captcha_id: str
    captcha_input: str

# RAG Data Schemas
class RagDataBase(BaseModel):
    name: str
    description: Optional[str] = None
    owner_id: Optional[int] = None

class RagDataCreate(RagDataBase):
    pass # No extra fields needed for creation beyond base

class RagData(RagDataBase):
    id: int
    is_active: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    access_level: Optional[str] = None # Add user's access level for this item

    class Config:
        from_attributes = True

# Response schema for listing RAG data
class RagDataListResponse(BaseModel):
    rag_data: List[RagData]
    can_create: bool

# Role Schemas
class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None

class RoleCreate(RoleBase):
    pass # No extra fields needed for creation beyond base

class Role(RoleBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    # The 'permissions' field is removed as permissions are now decoupled from roles
    # and handled by ABAC policies.

    class Config:
        from_attributes = True

class RoleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class RagDataUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[int] = None

# Schema for file upload related to RAG
class RagFileUpload(BaseModel):
    rag_id: int
    file_name: str
    # Add other relevant fields if needed, e.g., file_size, mime_type

class FileGist(BaseModel):
    id: int
    filename: str
    gist: Optional[str] = None
    created_at: datetime
    rag_id: int
    download_url: Optional[str] = None

    class Config:
        from_attributes = True

# User-Role Association Schema
class UserRoleAssign(BaseModel):
    user_id: int
    role_id: int


class FileEmbedRequest(BaseModel):
    file_ids: List[int]

class AgentChatRequest(BaseModel):
    message: str
    display_thoughts: Optional[bool] = False
    search_ai_active: Optional[bool] = True
    search_rag_active: Optional[bool] = False
    search_online_active: Optional[bool] = False
    files: Optional[List[UploadFile]] = None # This field will be handled by FastAPI's Form/File

# Policy Schemas
class AttributeFilter(BaseModel):
    key: str
    operator: str
    value: List[str]

class QueryCondition(BaseModel):
    resource_attribute: str
    operator: str
    subject_attribute: str

class PolicyBase(BaseModel):
    name: str
    description: Optional[str] = None
    effect: str = 'allow'
    actions: List[str]
    subjects: List[AttributeFilter]
    resources: List[AttributeFilter]
    query_conditions: Optional[List[QueryCondition]] = None
    is_active: Optional[int] = 1

class PolicyCreate(PolicyBase):
    pass

class PolicyUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    actions: Optional[List[str]] = None
    subjects: Optional[List[AttributeFilter]] = None
    resources: Optional[List[AttributeFilter]] = None
    query_conditions: Optional[List[QueryCondition]] = None
    is_active: Optional[int] = None
    effect: Optional[str] = None

class Policy(PolicyBase):
    id: int
    effect: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Schemas for ABAC permission checking
class CheckPermissionRequest(BaseModel):
    action: str
    resource_type: str
    resource_id: Optional[str] = None

class CheckPermissionResponse(BaseModel):
    allowed: bool

# Schema for ABAC attribute vocabulary
class AttributeSchema(BaseModel):
    key: str
    name: str
    category: str
    type: str