# `services/ollama_service.py` - Ollama Kommentar

Hinweis`backend/app/services/ollama_service.py` Hinweis

## Kommentar
*   **Ollama API Kommentar**: KommentaröschenKommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`ollama_module.py` MittelHinweis

Hinweis
```python
import httpx
from backend.app.core.config import settings
from backend.app.modules.ollama_module import OLLAMA_BASE_URL
import logging

logger = logging.getLogger(__name__)

async def get_ollama_models() -> list[dict]:
    """Lists available models from Ollama service."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Ollama service to list models: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing Ollama models: {e}")
        return []

async def chat_with_ollama(messages: list[dict], model_name: str) -> str:
    """
    Interacts with a specified Ollama model for chat completions.
    Messages should be in the format: [{"role": "user", "content": "..."}]
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False
                },
                timeout=600.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Ollama service for chat: {e}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama chat service returned an error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ollama chat: {e}")
        raise
```

## Kommentar
`/backend/app/services/ollama_service.py`