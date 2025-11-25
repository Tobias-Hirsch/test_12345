import os
import logging
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, func, 
    ForeignKey, Table, UniqueConstraint, JSON, Boolean, text, select, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from app.core.config import settings

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Database Connection ---
MYSQL_HOST = settings.MYSQL_HOST
MYSQL_PORT = settings.MYSQL_PORT
MYSQL_DB = settings.MYSQL_DB
MYSQL_USER = settings.MYSQL_USER
MYSQL_PASSWORD = settings.MYSQL_PASSWORD

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL, pool_recycle=3600, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Association Tables ---
user_role_association = Table(
    'user_role_association',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)


# --- ORM Models ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    phone = Column(String(255), unique=True, index=True, nullable=True)
    department = Column(String(100), nullable=True)
    security_level = Column(Integer, nullable=True)
    first_name = Column(String(100), nullable=True)
    surname = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    avatar = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    password_reset_token = Column(String(255), nullable=True, index=True)
    password_reset_expires_at = Column(DateTime, nullable=True)
    activation_token = Column(String(255), nullable=True, index=True)
    activation_expires_at = Column(DateTime, nullable=True)
    provider = Column(String(50), nullable=True)
    provider_id = Column(String(255), nullable=True)
    roles = relationship("Role", secondary="user_role_association", back_populates="users")
    is_verified = Column(Boolean, server_default='0', nullable=False)
    __table_args__ = (UniqueConstraint('provider', 'provider_id', name='_provider_provider_id_uc'),)

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    users = relationship("User", secondary="user_role_association", back_populates="roles")

class RagData(Base):
    __tablename__ = "rag_data"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Integer, default=1)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    owner = relationship("User")
    files = relationship("FileGist", back_populates="rag_data", cascade="all, delete-orphan")

class FileGist(Base):
    __tablename__ = "file_gists"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    gist = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    rag_id = Column(Integer, ForeignKey("rag_data.id"), nullable=False)
    rag_data = relationship("RagData", back_populates="files")
    file_hash = Column(String(64), nullable=True, unique=True, index=True, comment="SHA256 hash of the file content")
    
    # New fields for tracking processing status
    processing_status = Column(String(50), default='pending', nullable=False, comment="e.g., pending, processing, success, failed")
    processing_details = Column(Text, nullable=True, comment="To store error messages or other details")

    # Fields for third-party MinIO sync
    etag = Column(String(255), nullable=True, comment="ETag of the file from MinIO for version tracking")
    is_third_party = Column(Boolean, default=False, nullable=False, comment="Flag to indicate if the file is from a third-party source")
    
class Policy(Base):
    __tablename__ = "policies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    effect = Column(String(10), nullable=False, default="allow")
    actions = Column(JSON, nullable=False)
    subjects = Column(JSON, nullable=False)
    resources = Column(JSON, nullable=False)
    query_conditions = Column(JSON, nullable=True)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String(255), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=func.now())
    user = relationship("User")


# --- Utility Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_database_tables():
    """
    Create all database tables defined by SQLAlchemy models if they do not exist.
    Safe to call multiple times; it will only create missing tables.
    """
    logger.info("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database table creation complete.")
