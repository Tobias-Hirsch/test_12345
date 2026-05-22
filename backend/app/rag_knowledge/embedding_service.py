import numpy as np
import os
import httpx
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from ..core.config import settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings from the central config
OLLAMA_RERANKER_MODEL = settings.OLLAMA_RERANKER_MODEL
LOCAL_RERANKER_MODEL = settings.LOCAL_RERANKER_MODEL # For local development
LOCAL_EMBEDDING_MODEL = settings.LOCAL_EMBEDDING_MODEL
OLLAMA_DEVICE = settings.OLLAMA_DEVICE
EMBEDDING_STRATEGY = settings.EMBEDDING_STRATEGY
RERANK_STRATEGY = settings.RERANK_STRATEGY # 'local' or 'ollama'
OLLAMA_EMBEDDING_API_URL = settings.OLLAMA_EMBEDDING_API_URL
OLLAMA_RERANK_API_URL = settings.OLLAMA_RERANK_API_URL # New setting for the rerank API
OLLAMA_EMBEDDING_MODEL_NAME = settings.OLLAMA_EMBEDDING_MODEL_NAME


class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            if EMBEDDING_STRATEGY == 'local':
                try:
                    device = OLLAMA_DEVICE if OLLAMA_DEVICE else None
                    model_path = LOCAL_EMBEDDING_MODEL
                    logger.info(f"Initializing EmbeddingService with local model from path: {model_path}. Device: {device or 'auto'}")
                    cls._instance._model = SentenceTransformer(model_path, device=device)
                    logger.info(f"SentenceTransformer model from '{model_path}' loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading SentenceTransformer model: {e}", exc_info=True)
                    cls._instance._model = None
        return cls._instance

    def get_embedding_sync(self, text: str) -> Optional[np.ndarray]:
        """
        Synchronous method to generate an embedding for a single piece of text.
        """
        if self._model is None:
            logger.error("Embedding model is not loaded.")
            return None
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            return None

    def get_embeddings_sync(self, texts: List[str]) -> List[np.ndarray]:
        """
        Synchronous method to generate embeddings for a list of texts.
        """
        if self._model is None:
            logger.error("Embedding model is not loaded.")
            return []
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return []

def get_embedding_service():
    """Gets the singleton instance of the EmbeddingService."""
    return EmbeddingService()

async def get_embedding_from_ollama(text: str) -> Optional[np.ndarray]:
    """
    Asynchronously gets an embedding from the Ollama embedding service.
    """
    if not OLLAMA_EMBEDDING_API_URL or not OLLAMA_EMBEDDING_MODEL_NAME:
        logger.error("Ollama embedding API URL or model name is not configured.")
        return None
    
    payload = {
        "model": OLLAMA_EMBEDDING_MODEL_NAME,
        "prompt": text
    }
    
    try:
        # Larger chat models can temporarily block the Ollama queue.
        # Give embedding requests enough time to wait and complete.
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(OLLAMA_EMBEDDING_API_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding")
            
            if embedding:
                return np.array(embedding, dtype=np.float32)
            else:
                logger.error(f"Failed to get embedding from Ollama. Response: {data}")
                return None
    except httpx.RequestError as e:
        logger.error(f"HTTP request to Ollama embedding service failed: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while calling Ollama embedding service: {e}", exc_info=True)
        return None

async def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Asynchronously gets an embedding for a single text based on the configured strategy.
    """
    if EMBEDDING_STRATEGY == 'ollama':
        return await get_embedding_from_ollama(text)
    elif EMBEDDING_STRATEGY == 'local':
        service = get_embedding_service()
        # Run the synchronous blocking function in a separate thread
        return await asyncio.to_thread(service.get_embedding_sync, text)
    else:
        logger.error(f"Unknown embedding strategy: {EMBEDDING_STRATEGY}")
        return None


# --- New Reranker Implementation ---

# Global cache for local reranker model
_local_reranker_model = None

def _get_local_reranker():
    """Initializes and returns the local reranker model, cached globally."""
    global _local_reranker_model
    if _local_reranker_model is None:
        try:
            device = OLLAMA_DEVICE if OLLAMA_DEVICE else None
            model_name = LOCAL_RERANKER_MODEL
            logger.info(f"Initializing local reranker model: {model_name} on device: {device or 'auto'}")
            _local_reranker_model = CrossEncoder(model_name, max_length=512, device=device)
            logger.info("Local reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Fatal error loading local reranker model '{LOCAL_RERANKER_MODEL}': {e}", exc_info=True)
            # Set to a dummy object to prevent re-initialization attempts
            _local_reranker_model = "failed_to_load"
    
    return _local_reranker_model if _local_reranker_model != "failed_to_load" else None

async def _rerank_with_local_model(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reranks documents using a local CrossEncoder model."""
    model = _get_local_reranker()
    if not model or not documents:
        return documents

    pairs = [[query, doc.get("content", "")] for doc in documents]
    
    # Run blocking prediction in a thread pool
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        scores = await loop.run_in_executor(pool, model.predict, pairs)

    for i, doc in enumerate(documents):
        doc['rerank_score'] = scores[i]
    
    return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

async def _rerank_with_ollama(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reranks documents using the Ollama rerank API."""
    if not OLLAMA_RERANK_API_URL or not OLLAMA_RERANKER_MODEL:
        logger.error("Ollama rerank API URL or model name is not configured.")
        return documents

    # Extract just the content for the payload
    doc_contents = [doc.get("content", "") for doc in documents]

    payload = {
        "model": OLLAMA_RERANKER_MODEL,
        "query": query,
        "documents": doc_contents
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            logger.info(f"Sending {len(doc_contents)} documents to Ollama reranker: {OLLAMA_RERANKER_MODEL}")
            response = await client.post(OLLAMA_RERANK_API_URL, json=payload)
            response.raise_for_status()
            
            rerank_results = response.json().get("results", [])
            
            # The rerank API should return a list of {'document': str, 'relevance_score': float, 'index': int}
            # We need to map these scores back to our original document dictionaries.
            for result in rerank_results:
                original_index = result.get('index')
                if original_index is not None and original_index < len(documents):
                    documents[original_index]['rerank_score'] = result.get('relevance_score')

            # Sort by the new score, handling cases where some docs might not get a score
            return sorted(documents, key=lambda x: x.get('rerank_score', -float('inf')), reverse=True)

    except httpx.RequestError as e:
        logger.error(f"HTTP request to Ollama rerank service failed: {e}", exc_info=True)
        return documents # Return original documents on failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while calling Ollama rerank service: {e}", exc_info=True)
        return documents

async def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reranks documents based on the configured RERANK_STRATEGY.
    """
    if not documents:
        return []

    if RERANK_STRATEGY == 'ollama':
        return await _rerank_with_ollama(query, documents)
    elif RERANK_STRATEGY == 'local':
        return await _rerank_with_local_model(query, documents)
    else:
        logger.warning(f"Unknown or no RERANK_STRATEGY ('{RERANK_STRATEGY}'). Skipping reranking.")
        return documents


# Example Usage
async def main_test():
    # --- Test Reranker (Synchronous) ---
    print("--- Testing Reranker Service ---")
    reranker = get_reranker_service()
    if reranker.model:
        sample_query = "What is the capital of France?"
        sample_docs = [
            {'content': 'Paris is a beautiful city.'},
            {'content': 'The Eiffel Tower is in Paris.'},
            {'content': 'Berlin is the capital of Germany.'}
        ]
        reranked = rerank_documents(sample_query, sample_docs)
        print("\nReranked Documents:")
        for doc in reranked:
            print(f"Score: {doc['rerank_score']:.4f}, Content: {doc['content']}")
    else:
        print("Reranker model not loaded, skipping test.")

    # --- Test Embedding (Asynchronous) ---
    print("\n--- Testing Embedding Service ---")
    sample_text = "This is a test sentence for the embedding model."
    
    print(f"Current embedding strategy: '{EMBEDDING_STRATEGY}'")
    
    # Note: For 'ollama' strategy, ensure the service is running and configured in .env
    if EMBEDDING_STRATEGY == 'ollama':
        if not OLLAMA_EMBEDDING_API_URL or not OLLAMA_EMBEDDING_MODEL_NAME:
            print("Ollama config not set. Set OLLAMA_EMBEDDING_API_URL and OLLAMA_EMBEDDING_MODEL_NAME to test.")
            return
    
    embedding_vector = await get_embedding(sample_text)
    
    if embedding_vector is not None:
        print(f"Successfully generated embedding vector with shape: {embedding_vector.shape}")
        # print("Vector snippet:", embedding_vector[:5])
    else:
        print("Failed to generate embedding vector.")

if __name__ == "__main__":
    asyncio.run(main_test())
