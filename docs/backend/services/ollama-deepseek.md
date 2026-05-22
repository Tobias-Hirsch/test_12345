# `services/ollama_deepseek.py` - Ollama DeepSeek Kommentar

Hinweis`backend/app/services/ollama_deepseek.py` Hinweis

## Kommentar
*   **DeepSeek Kommentar**: Kommentar
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

async def chat_with_ollama_deepseek(messages: list[dict], model_name: str = "deepseek-coder") -> str:
    """
    Interacts with the DeepSeek model via Ollama for chat completions.
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
        logger.error(f"Error connecting to Ollama DeepSeek service: {e}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama DeepSeek service returned an error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ollama DeepSeek chat: {e}")
        raise
```

## Kommentar
`/backend/app/services/ollama_deepseek.py`