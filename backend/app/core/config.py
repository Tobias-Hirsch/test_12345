# Developer: Jinglu Han
# mailbox: admin@de-manufacturing.cn

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pydantic import field_validator
import os

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "Rosti"
    APP_VERSION: str = "0.1.1"
    TIMEZONE: str = "Asia/Shanghai"
    DESCRIPTION: str = "Rosti - An intelligent RAG platform"

    # Database settings
    MYSQL_HOST: str = "mysql"
    MYSQL_PORT: int = 3306
    MYSQL_DB: str = ""
    MYSQL_USER: str = ""
    MYSQL_PASSWORD: str = ""

    # Milvus settings
    MILVUS_HOST: str = ""
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "ai_collection"  # Default collection name for Milvus
    # MongoDB settings
    MONGO_URI: str = ""
    MONGO_DB_NAME: str = ""
    MONGO_AI_CHAT_HISTORY_COLLECTION: str = ""

    # Ollama settings
    OLLAMA_EMBEDDING_URL: str = ""
    OLLAMA_SERVING_URL: str = ""
    OLLAMA_RERANKER_MODEL: str = ""

    # Legacy Ollama settings used by backend/app/modules/ollama_module.py.
    # Keeping them here prevents AttributeError when the optional model-management
    # module is imported in local deployments.
    OLLAMA_HOST: str = "host.docker.internal"
    OLLAMA_PORT: int = 11434
    OLLAMA_MODEL: str = "qwen3:0.6b"
    OLLAMA_EMBEDDING_MODEL: str = "mxbai-embed-large:latest"
    OLLAMA_RERANK_MODEL: str = ""

    LOCAL_EMBEDDING_MODEL: str = ""
    LOCAL_RERANKER_MODEL: str = ""
    OLLAMA_QWEN_MODEL: str = ""
    OLLAMA_CHAT_MODEL: str = ""
    OLLAMA_COT_MODEL: str = ""
    OLLAMA_LLAVA_MODEL: str = ""
    OLLAMA_NUM_PREDICT: int = 4096
    OLLAMA_DEVICE: str = ""
    OLLAMA_QWEN_VL_MAX_LATEST: str = "qwen-vl-max"  # Default model for Qwen VL Max

    VECTOR_DIM: int = 1024

    # Embedding settings
    EMBEDDING_STRATEGY: str = "local"  # Can be "local" or "ollama"
    OLLAMA_EMBEDDING_API_URL: str = ""
    OLLAMA_EMBEDDING_MODEL_NAME: str = ""
    # RERANK settings
    RERANK_STRATEGY: str = "local"  # Can be "local" or "ollama"
    OLLAMA_RERANK_API_URL: str = ""  # New setting for the rerank API
    OLLAMA_RERANKER_MODEL_NAME: str = ""  


    # Deepseek API Key
    QW_API_KEY: str = ""

    # File path configuration
    PDF_PATH: str = ""
    WORD_PATH: str = ""
    CSV_PATH: str = ""
    IMG_PATH: str = ""
    MD_PATH: str = ""

    # PDF Parsing Strategy
    PDF_PARSE_STRATEGY: str = "pymupdf"
    PDF_PROCESSING_STRATEGY: str = "smart"  # Smart mode: auto-select based on environment and time
    MINERU_PARSE_BACKEND: str = "smart"  # Smart backend selection
    
    # Smart MinerU Configuration
    ENVIRONMENT: str = "production"  # production, development, test
    MINERU_NIGHTTIME_HOURS: str = "22-6"  # Night time hours for production
    MINERU_SGLANG_SERVER_URL: str = "http://1.116.119.85:8908"  # SGLang server (overridden by .env)
    
    # Optional: Force specific mode (for debugging)
    MINERU_FORCE_MODE: str = ""  # sglang, local-vlm, pipeline, or empty for auto
    
    PADDLEOCR_API_URL: str = ""

    # MinIO settings
    CUSTOMER_MINIO_ENDPOINT: str = ""
    CUSTOMER_MINIO_ACCESS_KEY: str = ""
    CUSTOMER_MINIO_SECRET_KEY: str = ""
    CUSTOMER_MINIO_BUCKET_NAME: str = ""
    MINIO_ENDPOINT: str = ""
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET_NAME: str = ""
    MINIO_CHAT_BUCKET_NAME: str = ""
    MINIO_AVATAR_BUCKET_NAME: str = "avatars"  # Default bucket for avatars

    # MinIO Sync Task Settings
    MINIO_SYNC_ENABLED: bool = False
    MINIO_SYNC_CRON_START_HOUR: str = "22" # Start hour for the sync window (e.g., 22 for 10 PM)
    MINIO_SYNC_CRON_END_HOUR: str = "6"   # End hour for the sync window (e.g., 6 for 6 AM)
    MINIO_SYNC_CRON_MINUTE: str = "0"    # Minute of the hour to run the job
    MINIO_SYNC_MAX_RETRIES: int = 3      # Default max retries for embedding
    MINIO_SYNC_FILES_PER_SUBDIR_LIMIT: int = 0 # 0 means no limit

    # SMTP settings
    SMTP_SERVER: str = ""
    SMTP_PORT: int = 465
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_SENDER_EMAIL: str = ""

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_USERNAME: str = ""
    REDIS_PASSWORD: str = ""
    REDIS_DATABASE: int = 2

    # File types
    DOCUMENT_TYPE_QWEN_VL_DEAL_IMAGE: List[str] = []

    # Bing Search settings
    BING_SEARCH_KEY: str = ""
    BING_SEARCH_ENDPOINT: str = ""

    # Bot Avatar URL
    BOT_AVATAR_URL: str = ""

    # Frontend URL for user activation
    FRONTEND_BASE_URL: str = "http://localhost"

    # Third-party authentication settings
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = ""
    GOOGLE_AUTH_ENABLE: bool = False

    MICROSOFT_CLIENT_ID: str = ""
    MICROSOFT_TENANT_ID: str = ""
    MICROSOFT_CLIENT_SECRET: str = ""
    MICROSOFT_REDIRECT_URI: str = ""
    MICROSOFT_AUTH_ENABLE: bool = False

    OAUTH_GENERIC_CLIENT_ID: str = ""
    OAUTH_GENERIC_CLIENT_SECRET: str = ""
    OAUTH_GENERIC_AUTHORIZATION_URL: str = ""
    OAUTH_GENERIC_TOKEN_URL: str = ""
    OAUTH_GENERIC_USERINFO_URL: str = ""
    OAUTH_GENERIC_SCOPE: str = ""
    OAUTH_GENERIC_REDIRECT_URI: str = ""
    OAUTH_GENERIC_AUTH_ENABLE: bool = False

    # JWT settings
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Multi-Agent configuration
    MAX_VALIDATION_RETRIES: int = 5

    # LDAP Configuration
    MSAD_SSO_ENABLED: bool = False
    LDAP_SERVER: str = ""
    LDAP_Domain: str = ""
    LDAP_BASE_DN: str = ""
    LDAP_BIND_DN: str = ""
    LDAP_BIND_PASSWORD: str = ""
    LDAP_USER_SEARCH_BASE: str = ""
    LDAP_USER_SEARCH_FILTER: str = ""
    LDAP_GROUP_SEARCH_BASE: str = ""
    LDAP_GROUP_SEARCH_FILTER: str = ""
    LDAP_USER_ID_ATTRIBUTE: str = ""
    LDAP_USER_DISPLAY_NAME_ATTRIBUTE: str = ""
    LDAP_USER_EMAIL_ATTRIBUTE: str = ""
    LDAP_USER_DEPARTMENT_ATTRIBUTE: str = ""
    LDAP_USER_SECURITY_LEVEL_ATTRIBUTE: str = ""
    LDAP_USER_ACCOUNT_CONTROL_ATTRIBUTE: str = ""
    LDAP_USE_SSL: bool = True
    LDAP_VALIDATE_CERT: bool = True

    # Conversation settings
    REMOVE_OLD_CONVERSATIONS_AFTER_DAYS: int = 30
    MAX_CONVERSATION_HISTORY_MESSAGES: int = 10
    MAX_CONTEXT_TOKENS: int = 6000 # Max tokens for the context window, leaving space for the response
    SCAN_DETECTION_THRESHOLD_DEFAULT: int = 20 # PDF scan detection character threshold, used when Redis is unavailable
    AI_TEMPLATE_SEGMENT_SPLIT_MAX_SIZE: int = 2000  # Default value, can be overridden in .env

    # Text Processing Settings
    TEXT_SPLITTER_STRATEGY: str = "recursive" # Can be "recursive" or "sentence"
    TEXT_CHUNK_SIZE: int = 1500
    TEXT_CHUNK_OVERLAP: int = 100
    LONG_TEXT_THRESHOLD: int = 2500 # Threshold to trigger long text summarization
      
    # Agentic configuration
    AGENTIC_RAG_ENABLE: bool = False

    # ABAC settings
    ABAC_POLICY_CACHE_EXPIRE_MINUTES: int = 60

    # Default Admin User
    CREATE_DEFAULT_ADMIN: bool = True
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_EMAIL: str = "admin@example.com"
    DEFAULT_ADMIN_PASSWORD: str = "change_this_password"

    # Logging settings
    LOG_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra="ignore")

settings = Settings()
