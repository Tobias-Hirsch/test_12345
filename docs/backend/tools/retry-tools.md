# `tools/retry_tools.py` - Kommentar

Hinweis`backend/app/tools/retry_tools.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentarührt ausKommentar
*   **Kommentar**: Kommentar
*   **FehlerKommentar**: Kommentar

## Kommentar
Hinweis`tenacity` Hinweis

Hinweis
```python
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, delay_seconds: int = 1, backoff_factor: int = 2, exceptions=(Exception,)):
    """
    A decorator to retry a function call if it fails.

    Args:
        max_attempts (int): Maximum number of attempts.
        delay_seconds (int): Initial delay between retries in seconds.
        backoff_factor (int): Factor by which the delay increases each time.
        exceptions (tuple): A tuple of exceptions to catch and retry on.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay_seconds
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}")
                    if attempt < max_attempts:
                        logger.info(f"Retrying {func.__name__} in {current_delay} seconds...")
                        await asyncio.sleep(current_delay) # For async functions
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}.")
                        raise
        return wrapper
    return decorator

# Kommentar
# @retry(max_attempts=5, delay_seconds=2, exceptions=(SomeAPIError, AnotherError))
# async def call_external_api():
#     # ... API Kommentar
#     pass
```

## Kommentar
`/backend/app/tools/retry_tools.py`