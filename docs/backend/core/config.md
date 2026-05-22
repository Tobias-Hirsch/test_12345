# `core/config.py` - Kommentar

Hinweis`backend/app/core/config.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **TypKommentar**: Kommentar`BaseSettings` OderKommentar
*   **Kommentar**: KommentarägeKommentar

## Kommentar
Hinweis`BaseSettings`(Hinweisäge, Hinweis`Field` OderHinweis

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "rosti-rag-files"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_COLLECTION_NAME: str = "rosti_rag_collection"
    MILVUS_VECTOR_DIM: int = 1536 # For OpenAI embeddings
    AGENTIC_RAG_ENABLE: bool = False
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: str
    SMTP_USER: str
    SMTP_PASSWORD: str
    EMAILS_FROM_EMAIL: str
    EMAILS_FROM_NAME: str = "Rosti Support"
    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_ACTIVATE_TOKEN_EXPIRE_HOURS: int = 48
    FIRST_SUPERUSER_EMAIL: str
    FIRST_SUPERUSER_PASSWORD: str
    FIRST_SUPERUSER_USERNAME: str = "admin"
    FIRST_SUPERUSER_FULL_NAME: str = "Admin User"
    CONVERSATION_CLEANUP_DAYS: int = 30 # Days after which old conversations are removed

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

Hinweis`Settings` Hinweis

## Kommentar
`/backend/app/core/config.py`