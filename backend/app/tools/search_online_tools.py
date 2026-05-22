import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from azure.cognitiveservices.search.websearch import WebSearchClient
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials
import os
from ..core.config import settings  # Global import for configuration settings  

# Replace with your Bing Search V7 subscription key and endpoint
# You should store this in environment variables or a configuration file
BING_SEARCH_KEY = settings.BING_SEARCH_KEY
BING_SEARCH_ENDPOINT = settings.BING_SEARCH_ENDPOINT

async def bingsearch(queries: List[str], max_results: int = 10) -> Dict[str, List[Any]]:
    """
    Hinweis

    Args:
        queries: Hinweis
        max_results: Hinweis

    Returns:
        Hinweis
    """
    results = {}

    if not BING_SEARCH_KEY or BING_SEARCH_KEY == "YOUR_BING_SEARCH_KEY":
        print("Bing Search API key not configured.")
        return {query: [] for query in queries}

    if not BING_SEARCH_ENDPOINT or BING_SEARCH_ENDPOINT == "YOUR_BING_SEARCH_ENDPOINT":
         print("Bing Search API endpoint not configured.")
         return {query: [] for query in queries}


    client = WebSearchClient(endpoint=BING_SEARCH_ENDPOINT, credentials=CognitiveServicesCredentials(BING_SEARCH_KEY))

    async def search_single_query(query: str) -> List[Any]:
        try:
            # Perform the search
            web_data = client.web.search(query=query, count=max_results, safe_search=SafeSearch.strict)

            # Process and return results
            bing_results = []
            if web_data.web_pages and web_data.web_pages.value:
                for item in web_data.web_pages.value:
                    bing_results.append({
                        "title": item.name,
                        "href": item.url,
                        "body": item.snippet
                    })
            return bing_results
        except Exception as e:
            print(f"Bing search for query '{query}' failed: {e}")
            return [] # Return empty list on failure

    # Process queries concurrently
    tasks = [search_single_query(query) for query in queries]
    all_results = await asyncio.gather(*tasks)

    # Associate results with queries
    for query, result in zip(queries, all_results):
        results[query] = result

    return results

async def duckduckgosearch(queries: List[str], max_results: int = 10, max_retries: int = 3, retry_delay: float = 2.0) -> \
Dict[str, List[Any]]:
    """
    Hinweis

    Args:
        queries: Hinweis
        max_results: Hinweis
        max_retries: Hinweis
        retry_delay: Hinweis

    Returns:
        Hinweis
    """
    results = {}

    async def search_single_query(query: str) -> List[Any]:
        loop = asyncio.get_event_loop()

        for attempt in range(max_retries):
            try:
                with ThreadPoolExecutor() as executor:
                    return await loop.run_in_executor(
                        executor,
                        lambda: list(DDGS().text(query, max_results=max_results))
                    )
            except DuckDuckGoSearchException as e:
                if "Ratelimit" in str(e) and attempt < max_retries - 1:
                    # Kommentar
                    jitter = random.uniform(0.3, 0.5)
                    wait_time = retry_delay * (2 ** attempt) * jitter
                    print(f"Abfragen '{query}' Hinweis{wait_time:.2f} Hinweis{attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Abfragen '{query}' Hinweis{e}")
                    return []  # Hinweis

        return []  # Hinweis

    # Kommentar
    batch_size = 2  # Hinweis
    all_results = []

    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]

        # Kommentar
        tasks = [search_single_query(query) for query in batch_queries]

        # Kommentar
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

        # Kommentar
        if i + batch_size < len(queries):
            await asyncio.sleep(1.0)  # Hinweis

    # Kommentar
    for query, result in zip(queries, all_results):
        results[query] = result

    return results
