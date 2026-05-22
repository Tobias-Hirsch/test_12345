# `llm/chain.py` - LLM Kommentar

Hinweis`backend/app/llm/chain.py` Hinweisührt aus LLM Hinweis

## Kommentar
*   **LLM Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis

Hinweis
```python
from typing import Optional
from backend.app.llm.llm import get_llm_client
import logging

logger = logging.getLogger(__name__)

async def fn_async_summarize_law(text: str, query: Optional[str] = None) -> str:
    """
    Asynchronously summarizes legal text using an LLM.
    Optionally takes a query to focus the summary.
    """
    llm = get_llm_client() # Hinweis

    system_prompt = "Anweisung"
    if query:
        system_prompt += f"Anweisung'{query}'. "

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Hinweis\n\n{text}"}
    ]

    try:
        response = await llm.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error summarizing law with LLM: {e}")
        raise
```

## Kommentar
`/backend/app/llm/chain.py`