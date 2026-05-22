# `services/logging.py` - Kommentar

Hinweis`backend/app/services/logging.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`logging` Hinweis

Hinweis
```python
import logging
import os

def configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(), # Hinweis
            # logging.FileHandler("app.log") # Kommentar
        ]
    )
    # Kommentar
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Kommentar
# configure_logging()
```

## Kommentar
`/backend/app/services/logging.py`