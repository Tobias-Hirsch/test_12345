# `llm/llm.py` - LLM Kommentar

Hinweis`backend/app/llm/llm.py` Hinweis

## Kommentar
*   **LLM Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis

Hinweis
```python
import os
from backend.app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Kommentar
_llm_client = None

def get_llm_client():
    """
    Returns an initialized LLM client based on configuration.
    Supports OpenAI and Ollama.
    """
    global _llm_client
    if _llm_client:
        return _llm_client

    if settings.OPENAI_API_KEY:
        try:
            from openai import OpenAI
            _llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            _llm_client.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo") # Hinweis
            logger.info(f"Initialized OpenAI LLM client with model: {_llm_client.model}")
            return _llm_client
        except ImportError:
            logger.error("OpenAI library not found. Please install it: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM client: {e}")
            raise
    elif settings.OLLAMA_BASE_URL and settings.OLLAMA_CHAT_MODEL:
        try:
            from openai import OpenAI # OllamaHinweis
            _llm_client = OpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama" # Ollama API key is typically not required or can be a placeholder
            )
            _llm_client.model = settings.OLLAMA_CHAT_MODEL
            logger.info(f"Initialized Ollama LLM client with base URL: {settings.OLLAMA_BASE_URL} and model: {_llm_client.model}")
            return _llm_client
        except ImportError:
            logger.error("OpenAI library not found. Please install it: pip install openai (required for Ollama compatibility)")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM client: {e}")
            raise
    else:
        raise ValueError("No LLM provider configured. Please set OPENAI_API_KEY or OLLAMA_BASE_URL/OLLAMA_CHAT_MODEL.")

```

## Kommentar
`/backend/app/llm/llm.py`