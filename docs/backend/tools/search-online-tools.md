# `tools/search_online_tools.py` - Kommentar

Hinweis`backend/app/tools/search_online_tools.py` Hinweisührt ausHinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **ErgebnisseKommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`httpx` Oder `requests` Hinweis

Hinweis
```python
import httpx
import os
import json
import logging

logger = logging.getLogger(__name__)

# Kommentar
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

async def bing_web_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Performs a web search using Bing Search API.
    Returns a list of dictionaries, each representing a search result.
    """
    if not BING_SEARCH_API_KEY:
        logger.error("BING_SEARCH_API_KEY is not set in environment variables.")
        return []

    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
    params = {"q": query, "count": num_results}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(BING_SEARCH_ENDPOINT, headers=headers, params=params)
            response.raise_for_status() # Raises an HTTPStatusError for bad responses (4xx or 5xx)
            search_results = response.json()
            
            results = []
            if "webPages" in search_results:
                for item in search_results["webPages"]["value"]:
                    results.append({
                        "name": item.get("name"),
                        "url": item.get("url"),
                        "snippet": item.get("snippet")
                    })
            logger.info(f"Bing search for '{query}' returned {len(results)} results.")
            return results
    except httpx.RequestError as e:
        logger.error(f"An error occurred while requesting Bing Search API: {e}")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"Bing Search API returned an error: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during Bing search: {e}")
        return []

# Kommentar
# async def main():
#     results = await bing_web_search("latest AI news")
#     for r in results:
#         print(f"Name: {r['name']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n---")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
```

## Kommentar
`/backend/app/tools/search_online_tools.py`