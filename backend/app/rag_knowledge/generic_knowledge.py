from datetime import datetime
import os
import re
import uuid
import asyncio
import base64
import io
import logging
import json
from typing import List, Dict, Any, Tuple, Optional, AsyncGenerator
import aiofiles
import numpy as np
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import markdown
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
from sqlalchemy.orm import Session
from fastapi import UploadFile
from minio.error import S3Error
from datetime import timedelta

from app.models.database import FileGist
from app.schemas import schemas
from app.services.file_upload_service import get_minio_client
from ..core.config import settings # Global import
from app.services.mineru_parser import parse_doc # Import our new parser
from app.core.state import doc_processing_lock # Import the global lock

# Configure logger
logger = logging.getLogger(__name__)


MILVUS_HOST = settings.MILVUS_HOST
MILVUS_PORT = settings.MILVUS_PORT
# OLLAMA_URL is deprecated, will be replaced by specific URLs later
MONGO_URI = settings.MONGO_URI
MONGO_DB_NAME= settings.MONGO_DB_NAME
PDF_PATH = settings.PDF_PATH
WORD_PATH = settings.WORD_PATH
CSV_PATH = settings.CSV_PATH
IMG_PATH = settings.IMG_PATH
MD_PATH = settings.MD_PATH
bucket_name = settings.MINIO_BUCKET_NAME
OLLAMA_DEVICE= settings.OLLAMA_DEVICE  # Default to mps:0 if not set
# Import the LLM object
from app.llm.llm import get_llm

# Kommentar
from app.rag_knowledge.embedding_service import get_embedding, rerank_documents
from app.services import ollama_service # Import the central service
from app.services.paddleocr_service import call_paddleocr_service
from app.services.latexocr_service import call_latexocr_service
from transformers import AutoProcessor, AutoModelForImageTextToText
from app.tools.pdf import process_pdf_with_mineru
from app.tools.deal_document import summary_documents_content
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.modules.minio_module import store_json_object_in_minio


async def _process_image_with_paddle(file_bytes: bytes, filename: str) -> str:
    """
    Helper function to process a single image file with the lightweight PaddleOCR service.
    (Copied from document_processing_service to consolidate logic)
    """
    logger.info(f"Processing image '{filename}' with lightweight PaddleOCR service.")
    file_like_object = io.BytesIO(file_bytes)
    upload_file = UploadFile(filename=filename, file=file_like_object)
    
    # The lightweight service returns a dictionary with a "text" key.
    paddle_result = await call_paddleocr_service(upload_file)
    
    if not paddle_result or 'text' not in paddle_result:
        logger.error(f"Lightweight PaddleOCR processing failed for '{filename}'.")
        return ""
        
    return paddle_result.get('text', '')


def ocr_image(image_path: str):
    device = OLLAMA_DEVICE 
    model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
    processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

    try:
        # Open the image file using Pillow
        image = Image.open(image_path)
        inputs = processor(image, return_tensors="pt").to(device)

        generate_ids = model.generate(
            **inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=4096,
        )
        wor = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        logger.info(wor)
        return wor
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True)
        raise # Re-raise the exception after printing

# Kommentar
async def connect_to_milvus(host: Optional[str] = None, port: Optional[str] = None):
    """Hinweis"""
    # Use passed host/port if available, otherwise use environment variables,
    # otherwise use hardcoded defaults.
    milvus_host = host if host is not None else MILVUS_HOST
    milvus_port = port if port is not None else MILVUS_PORT

    logger.info(f"Attempting to connect to Milvus at {milvus_host}:{milvus_port}")
    try:
        connections.connect(
            host=milvus_host,
            port=milvus_port,
            timeout=30.0,  # Hinweis
            wait_timeout=30  # Hinweis
        )
        logger.info("Hinweis")
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung{str(e)}", exc_info=True)
        raise


# Kommentar
async def create_collection(collection_name: str = "markdown_data", dim: int = 1024):
    """Hinweis"""
    # if Collection(collection_name).exists():
    #     return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="is_image", dtype=DataType.VARCHAR, max_length=20)
    ]

    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)

    # Kommentar
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info(f"Hinweis{collection_name}")
    return collection


async def read_markdown_file(file_path: str) -> str:
    """Hinweis"""
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        content = await f.read()
    return content


async def parse_markdown(content: str, base_path: str, image_base_path: str) -> List[Dict[str, Any]]:
    """Hinweis"""
    # Kommentar
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')

    chunks = []

    # Kommentar
    # Expand the list of tags to extract text from, ensuring comprehensive content capture.
    # This now includes all header levels, paragraphs, list items, and table cells.
    for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
        if element.text.strip():
            chunks.append({
                "content": element.text.strip(),
                "is_image": "False",
                "filepath": base_path
            })

    # Kommentar
    for img in soup.find_all('img'):
        img_src = img.get('src', '')
        if img_src:
            logger.info(img_src)
            # Kommentar
            if not os.path.isabs(img_src):
                # Use the provided image_base_path for joining
                img_path = os.path.join(image_base_path, os.path.basename(img_src))
            else:
                img_path = img_src

            # Kommentar
            try:
                ocr_text = ocr_image(img_path)

                # Kommentar
                context = ""
                img_index = soup.find_all().index(img)

                # Kommentar
                for i in range(img_index - 1, -1, -1):
                    element = soup.find_all()[i]
                    # if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and element.text.strip():
                    if element.name in ['p', 'h1', 'h2' ] and element.text.strip():
                        context += element.text.strip() + " "
                        break

                # Kommentar
                for i in range(img_index + 1, len(soup.find_all())):
                    element = soup.find_all()[i]
                    # if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and element.text.strip():
                    if element.name in ['p', 'h1', 'h2'] and element.text.strip():
                        context += element.text.strip()
                        break

                # Kommentar+Kommentar
                img_content = f"{ocr_text} [Hinweis{context.strip()}]"

                chunks.append({
                    "content": img_content,
                    "is_image": "True",
                    "filepath": img_path
                })
            except Exception as e:
                logger.error(f"Fehler bei der Bildverarbeitung: {img_path}, Fehler: {str(e)}", exc_info=True)
    logger.debug(f"Parsed chunks: {chunks}")
    return chunks


async def process_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hinweis"""
    processed_chunks = []

    for chunk in chunks:
        # Kommentar
        chunk_id = str(uuid.uuid4())

        # Kommentar
        vector = await get_embedding(chunk["content"])
 
        # Check if embedding was successful before appending
        if vector is not None and vector.size > 0:
            processed_chunks.append({
                "id": chunk_id,
                "vector": vector,
                "filepath": chunk["filepath"],
                "content": chunk["content"],
                "is_image": chunk["is_image"]
            })
        else:
            logger.warning(f"Skipping chunk due to failed embedding: {chunk['filepath']}")

    return processed_chunks


async def insert_to_milvus(collection: Collection, data: List[Dict[str, Any]]):
    """Hinweis"""
    if not data:
        return

    # Kommentar
    ids = [item["id"] for item in data]
    vectors = [item["vector"] for item in data]
    filepaths = [item["filepath"] for item in data]
    contents = [item["content"] for item in data]
    is_images = [item["is_image"] for item in data]

    # Kommentar
    # if not all(isinstance(vec, list) and all(isinstance(x, float) for x in vec) for vec in vectors):
    #     raise ValueError("Vectors must be a list of lists of floats")

    # Kommentar
    collection.insert([ids, vectors, filepaths, contents, is_images])
    logger.info(f"Hinweis{len(ids)} Hinweis")

    # Flush and load the collection to make data searchable immediately
    logger.info("Flushing collection...")
    collection.flush()
    logger.info("Loading collection...")
    collection.load()
    logger.info("Collection loaded and ready for search.")


async def delete_milvus_data_by_filepath(collection_name: str, filepath: str):
    """Deletes Milvus entities based on the filepath."""
    await connect_to_milvus()
    try:
        collection = Collection(collection_name)
        # Ensure the collection is loaded before performing delete operations
        collection.load()
        # Use a delete expression to remove entities with the matching filepath
        # Note: String fields require double quotes in the expression
        delete_expr = f'filepath == "{filepath}"'
        result = collection.delete(delete_expr)
        logger.info(f"Deleted Milvus entities with filepath '{filepath}': {result}")
    except Exception as e:
        logger.error(f"Error deleting Milvus data for filepath '{filepath}': {e}", exc_info=True)
        # Depending on requirements, you might want to raise the exception or handle it differently


async def delete_mongo_data_by_filename(mongo_db_name: str, mongo_collection_name: str, filename: str):
    """Deletes MongoDB document metadata based on the original filename."""
    mongo_client = get_mongo_client()
    if not mongo_client:
        logger.error("Failed to connect to MongoDB. Cannot delete document metadata.")
        return

    try:
        db = mongo_client[mongo_db_name]
        documents_collection = db[mongo_collection_name]
        # Delete the document where original_filename matches
        result = documents_collection.delete_one({"original_filename": filename})
        logger.info(f"Deleted MongoDB document with filename '{filename}': {result.deleted_count} document(s)")
    except Exception as e:
        logger.error(f"Error deleting MongoDB data for filename '{filename}': {e}", exc_info=True)
        # Depending on requirements, you might want to raise the exception or handle it differently
    finally:
        if mongo_client:
            mongo_client.close()


async def search_in_milvus(
    query_text: str,
    collection_name: str = "markdown_data",
    recall_limit: int = 10,
    filter_expr: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Hinweis"""
    await connect_to_milvus()
    collection = Collection(collection_name)
    collection.load()

    # Kommentar
    query_vector = await get_embedding(query_text)
    if query_vector is None:
        logger.error(f"Failed to generate embedding for query: {query_text}")
        return []

    # führt ausSuchen
    search_params = {"metric_type": "L2", "params": {"ef": 64}}
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=recall_limit,
        expr=filter_expr,  # Apply the metadata filter here
        output_fields=["filepath", "content", "is_image"]
    )

    # Kommentar
    search_results = []
    for hits in results:
        for hit in hits:
            search_results.append({
                "id": hit.id,
                "distance": hit.distance,
                "filepath": hit.entity.get("filepath"),
                "content": hit.entity.get("content"),
                "is_image": hit.entity.get("is_image")
            })

    # Return all results up to the recall_limit for reranking
    return search_results


def _normalize_document_summary(summary: Any) -> str:
    """Filters technical artifacts out of stored document summaries."""
    if not isinstance(summary, str):
        return "No reliable summary available."

    cleaned_summary = " ".join(summary.split()).strip()
    if not cleaned_summary:
        return "No reliable summary available."

    invalid_markers = (
        "multiple items of type",
        "requires a single string output",
        "single string output",
        "format_instructions",
        "jsonoutputparser",
    )
    if any(marker in cleaned_summary.lower() for marker in invalid_markers):
        return "No reliable summary available."

    return cleaned_summary


async def retrieve_rag_context(
    query_text: str,
    db_session: Session,
    permitted_rag_items: Optional[List[Dict[str, Any]]] = None,
    limit: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieves raw RAG context and source metadata without asking the LLM.

    Chat response generation needs the original document chunks, not a
    pre-synthesized answer, otherwise a first-pass model can turn a valid
    retrieval result into an unhelpful refusal.
    """
    await connect_to_milvus()

    all_search_results = []

    if permitted_rag_items:
        for rag_item in permitted_rag_items:
            rag_item_id = rag_item.get("id")
            rag_item_name = rag_item.get("name")

            if not rag_item_id or not rag_item_name:
                continue

            # Keep this in sync with the embedding pipeline in routers/embeddings.py.
            sanitized_rag_item_name = rag_item_name.lower().replace(" ", "_")
            collection_name = f"rag_{sanitized_rag_item_name}"

            if not utility.has_collection(collection_name):
                logger.warning(f"Skipping RAG item '{rag_item_name}': Collection '{collection_name}' does not exist.")
                continue

            try:
                search_results = await search_in_milvus(query_text, collection_name=collection_name, recall_limit=50)

                for result in search_results:
                    result["rag_item_id"] = rag_item_id
                    result["rag_item_name"] = rag_item_name
                all_search_results.extend(search_results)

            except Exception as e:
                logger.error(f"Error querying RAG item '{rag_item_name}' (collection: {collection_name}): {e}", exc_info=True)

    if all_search_results:
        logger.info(f"--- Reranking: Starting with {len(all_search_results)} initial candidates. ---")

        reranked_results = await rerank_documents(query_text, all_search_results)
        all_search_results = reranked_results[:limit]

        logger.info(f"--- Reranking Complete: Top {len(all_search_results)} results selected. ---")
        for i, doc in enumerate(all_search_results):
            score_info = f"Score: {doc.get('relevance_score'):.4f}" if 'relevance_score' in doc else f"Dist: {doc.get('distance'):.4f}"
            logger.info(f"  [{i+1}] ID: {doc.get('id')}, {score_info}, Path: {doc.get('filepath')}")
            logger.info(f"      Content: {doc.get('content', '')[:300]}...")
        logger.info("-" * 50)

    mongo_client = get_mongo_client()
    if not mongo_client:
        logger.error("Failed to connect to MongoDB. Cannot retrieve additional context.")
        return "", []

    combined_context = []
    source_documents = []

    try:
        for result in all_search_results:
            chunk_id = result.get("id")
            rag_item_name = result.get("rag_item_name")
            if not rag_item_name:
                logger.warning(f"Skipping result due to missing rag_item_name: {result}")
                continue

            sanitized_rag_item_name = rag_item_name.lower().replace(" ", "_")
            mongo_db_name_for_rag = f"rag_db_{sanitized_rag_item_name}"
            mongo_collection_name_for_rag = f"documents_{sanitized_rag_item_name}"

            try:
                db = mongo_client[mongo_db_name_for_rag]
                documents_collection = db[mongo_collection_name_for_rag]

                if chunk_id:
                    document = documents_collection.find_one({"milvus_chunks.chunk_id": chunk_id})
                    if document:
                        original_filename = document.get('original_filename', 'N/A')
                        chunk_content = result.get('content', 'N/A')
                        summary = _normalize_document_summary(document.get('summary'))
                        context_text = f"From document '{original_filename}' (Summary: {summary}):\n---\nContent Chunk: {chunk_content}\n---"
                        combined_context.append(context_text)

                        logger.debug(f"--- DEBUG: Attempting to find download URL for filename: '{original_filename}' in RAG ID: {result.get('rag_item_id')} ---")
                        file_gist_with_url = get_file_gist_by_filename_and_rag_id(
                            db_session,
                            filename=original_filename,
                            rag_id=result.get("rag_item_id")
                        )

                        source_doc_item = {
                            "type": "RAG",
                            "rag_item_name": rag_item_name,
                            "source": original_filename,
                            "content_preview": chunk_content[:100] + '...' if len(chunk_content) > 100 else chunk_content,
                            "summary": summary
                        }

                        if file_gist_with_url:
                            logger.debug(f"  [SUCCESS] Found file_gist with ID: {file_gist_with_url.id} for filename '{original_filename}'")
                            source_doc_item["file_id"] = file_gist_with_url.id
                            source_doc_item["filename"] = file_gist_with_url.filename
                        else:
                            logger.warning(f"  [WARNING] Could not find a file_gist for '{original_filename}'.")

                        source_documents.append(source_doc_item)
            except Exception as e:
                logger.error(f"Error accessing MongoDB for RAG item '{rag_item_name}' (DB: {mongo_db_name_for_rag}, Collection: {mongo_collection_name_for_rag}): {e}", exc_info=True)
    finally:
        mongo_client.close()

    return "\n\n".join(combined_context), source_documents


async def query_rag_system(
    query_text: str,
    db_session: Session, # Add db_session parameter
    permitted_rag_items: Optional[List[Dict[str, Any]]] = None, # New parameter for permitted RAG items
    limit: int = 5,
    show_think_process: bool = False # Add this parameter
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Queries the RAG system with a user question, searching across permitted RAG items.

    Args:
        query_text: The user's query string.
        permitted_rag_items: Optional list of dictionaries, each containing 'id' and 'name'
                             for RAG items the user has access to. If None or empty,
                             no RAG data will be searched.
        limit: The maximum number of search results to return per RAG item.

    Returns:
        An async generator that yields the synthesized answer and source documents.
    """
    # Connect to Milvus
    await connect_to_milvus()

    all_search_results = []

    # If permitted_rag_items is provided and not empty, search across those collections
    if permitted_rag_items:
        for rag_item in permitted_rag_items:
            rag_item_id = rag_item.get("id")
            rag_item_name = rag_item.get("name")

            if not rag_item_id or not rag_item_name:
                continue

            # Construct the Milvus collection name based on RAG item name
            # Sanitize rag_item_name for Milvus collection name (replace spaces with underscores)
            sanitized_rag_item_name = rag_item_name.lower().replace(" ", "_")
            collection_name = f"rag_{sanitized_rag_item_name}"

            # Check if the collection exists before trying to query it
            if not utility.has_collection(collection_name):
                logger.warning(f"Skipping RAG item '{rag_item_name}': Collection '{collection_name}' does not exist.")
                continue

            try:
                # Search in this specific Milvus collection
                search_results = await search_in_milvus(query_text, collection_name=collection_name, recall_limit=50)

                # Add rag_item_id and name to the search results for context/frontend
                for result in search_results:
                    result["rag_item_id"] = rag_item_id
                    result["rag_item_name"] = rag_item_name
                all_search_results.extend(search_results)

            except Exception as e:
                logger.error(f"Error querying RAG item '{rag_item_name}' (collection: {collection_name}): {e}", exc_info=True)
                # Continue to the next permitted RAG item
    # Sort all results by distance (assuming search_in_milvus returns results with distance)
    # and take the top N overall results (e.g., top 10 or 20 overall)
    # This requires search_in_milvus to return distance and potentially more results than the final limit
    # Let's adjust search_in_milvus to return distance and query_rag_system to handle overall limit
    # For now, we'll just combine and return, assuming search_in_milvus handles its own limit per collection.
    # A more sophisticated approach would be to get more results per collection and re-rank globally.
    
    # --- Reranking Step ---
    if all_search_results:
        logger.info(f"--- Reranking: Starting with {len(all_search_results)} initial candidates. ---")
        
        # Call the new, flexible reranker service
        reranked_results = await rerank_documents(query_text, all_search_results)
        
        # Take the top 'limit' results after reranking
        all_search_results = reranked_results[:limit]

        logger.info(f"--- Reranking Complete: Top {len(all_search_results)} results selected. ---")
        for i, doc in enumerate(all_search_results):
            # Log score if available (from reranker), otherwise distance
            score_info = f"Score: {doc.get('relevance_score'):.4f}" if 'relevance_score' in doc else f"Dist: {doc.get('distance'):.4f}"
            logger.info(f"  [{i+1}] ID: {doc.get('id')}, {score_info}, Path: {doc.get('filepath')}")
            logger.info(f"      Content: {doc.get('content', '')[:300]}...")
        logger.info("-" * 50)


    mongo_client = get_mongo_client()
    if not mongo_client:
        logger.error("Failed to connect to MongoDB. Cannot retrieve additional context.")
        yield {"answer": "Fehler: Verbindung zur Metadatenbank nicht möglich. Bitte prüfen Sie die Systemkonfiguration.", "source_documents": []}
        return

    combined_context = []
    source_documents = []

    for result in all_search_results:
        chunk_id = result.get("id")
        rag_item_name = result.get("rag_item_name")
        if not rag_item_name:
            logger.warning(f"Skipping result due to missing rag_item_name: {result}")
            continue

        # Keep this in sync with the embedding pipeline in routers/embeddings.py.
        sanitized_rag_item_name = rag_item_name.lower().replace(" ", "_")
        mongo_db_name_for_rag = f"rag_db_{sanitized_rag_item_name}"
        mongo_collection_name_for_rag = f"documents_{sanitized_rag_item_name}"

        try:
            db = mongo_client[mongo_db_name_for_rag]
            documents_collection = db[mongo_collection_name_for_rag]

            if chunk_id:
                # Find the document in the specific MongoDB collection that contains this chunk ID
                document = documents_collection.find_one({"milvus_chunks.chunk_id": chunk_id})
                if document:
                    original_filename = document.get('original_filename', 'N/A')
                    chunk_content = result.get('content', 'N/A')

                    # Add document summary and chunk content to the context for LLM
                    # Combine the document summary and the specific chunk content for richer context.
                    summary = _normalize_document_summary(document.get('summary'))
                    context_text = f"From document '{original_filename}' (Summary: {summary}):\n---\nContent Chunk: {chunk_content}\n---"
                    combined_context.append(context_text)

                    # Retrieve file gist with download URL using the new service
                    # Use the new service function to get the file gist by filename and RAG item ID
                    logger.debug(f"--- DEBUG: Attempting to find download URL for filename: '{original_filename}' in RAG ID: {result.get('rag_item_id')} ---")
                    file_gist_with_url = get_file_gist_by_filename_and_rag_id(
                        db_session,
                        filename=original_filename, # Use the original_filename from Mongo directly
                        rag_id=result.get("rag_item_id")
                    )
                    
                    # Collect source information for frontend display
                    source_doc_item = {
                        "type": "RAG",
                        "rag_item_name": rag_item_name,
                        "source": original_filename,
                        "content_preview": chunk_content[:100] + '...' if len(chunk_content) > 100 else chunk_content,
                        "summary": summary # Add summary from MongoDB
                    }
                    
                    if file_gist_with_url:
                        logger.debug(f"  [SUCCESS] Found file_gist with ID: {file_gist_with_url.id} for filename '{original_filename}'")
                        # Replace download_url with file_id for the new secure download mechanism
                        source_doc_item["file_id"] = file_gist_with_url.id
                        source_doc_item["filename"] = file_gist_with_url.filename # Keep the full filename for display
                    else:
                        logger.warning(f"  [WARNING] Could not find a file_gist for '{original_filename}'.")
                    
                    source_documents.append(source_doc_item)
        except Exception as e:
            logger.error(f"Error accessing MongoDB for RAG item '{rag_item_name}' (DB: {mongo_db_name_for_rag}, Collection: {mongo_collection_name_for_rag}): {e}", exc_info=True)
            # Continue processing other results even if one MongoDB access fails


    # Close MongoDB connection
    mongo_client.close()

    # Synthesize Answer
    if not combined_context:
        synthesized_answer = "Auf Basis der bereitgestellten Kontextinformationen konnte ich keine passende Antwort finden."
        yield {"answer": synthesized_answer, "source_documents": source_documents}
        return

    # --- New, more effective prompt structure ---
    system_prompt = """Du bist ein professioneller und sorgfältiger Q&A-Assistent.
Deine Aufgabe ist es, die Frage des Benutzers **streng und ausschließlich** auf Grundlage des bereitgestellten Kontexts zu beantworten.

**Regeln:**
1. **Kontext analysieren:** Lies alle bereitgestellten Kontextabschnitte sorgfältig, bevor du eine Antwort formulierst.
2. **Strikte Kontextbindung:** Deine gesamte Antwort MUSS aus den Informationen innerhalb der `<context>`-Blöcke abgeleitet werden. Verwende KEIN externes Wissen und triff keine Annahmen.
3. **Direkte Antwort:** Beantworte die Frage direkt und umfassend.
4. **Umgang mit fehlenden Informationen:**
   * Wenn der Kontext genügend Informationen enthält, gib eine vollständige Antwort.
   * Wenn der Kontext irrelevant ist oder die benötigten Informationen nicht enthält, MUSST du exakt mit diesem Satz antworten: "Auf Grundlage der mir vorliegenden Materialien kann ich Ihre Frage nicht beantworten."
   * Rate nicht und gib keine Teilantwort, wenn die Information nicht ausdrücklich im Kontext enthalten ist.
"""

    user_prompt_template = """
<context>
{context}
</context>

<question>
{query}
</question>

Beantworte die Frage anhand der vorgegebenen Regeln ausschließlich auf Basis des obigen Kontexts.
"""
    
    user_prompt = user_prompt_template.format(context="\n\n".join(combined_context), query=query_text)

    # --- DEBUGGING: Print the final context and prompt ---
    logger.debug("--- DEBUG: System Prompt for LLM ---")
    logger.debug(system_prompt)
    logger.debug("--- DEBUG: User Prompt for LLM ---")
    logger.debug(user_prompt)
    # --- END DEBUGGING ---

    llm_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    try:
        # Use the centralized ollama_service to stream the response
        # Yield each chunk of the answer as it is received
        async for chunk in ollama_service.get_chat_response_stream(llm_messages, show_think_process=show_think_process):
            yield chunk
        
        # After the text stream is complete, yield the final metadata object
        # This object only contains the source documents.
        logger.debug("\n--- DEBUG: Yielding final source_documents metadata ---")
        logger.debug(source_documents)
        logger.debug("-" * 50)
        yield {"source_documents": source_documents}

    except Exception as e:
        logger.error(f"Error during LLM synthesis: {e}", exc_info=True)
        # In case of an error, yield an error message chunk
        yield f"Error synthesizing answer: {e}"
        # Also yield an empty source_documents object to finalize the frontend state
        yield {"source_documents": []}


def _clean_text(text: str) -> str:
    """Cleans the extracted text by removing garbled lines."""
    cleaned_lines = []
    for line in text.split('\n'):
        # This regex checks if a line has a very low ratio of alphanumeric characters (and common punctuation)
        # to total characters. This is a heuristic to detect garbled lines.
        if len(line.strip()) == 0:
            continue
        
        # Count valid characters (alphanumeric characters, CJK characters, common punctuation)
        valid_chars = re.findall(r'[\w\s\u4e00-\u9fff.,!?;:()"\'''""`~-]', line)
        
        # If the line is not empty and the ratio of valid characters is too low, consider it garbled.
        if len(line) > 0 and len(valid_chars) / len(line) < 0.3:
            logger.debug(f"  [Cleaning] Removing garbled line: {line}")
            continue
        
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def _parse_with_pymupdf(pdf_path: str) -> str:
    """Parses PDF using PyMuPDF, cleans the output, and returns all text content."""
    full_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text() + "\n"
        logger.info(f"Successfully parsed PDF with PyMuPDF: {pdf_path}")
        
        # Clean the extracted text before returning
        cleaned_text = _clean_text(full_text)
        return cleaned_text
        
    except Exception as e:
        logger.error(f"[ERROR] PyMuPDF failed to parse {pdf_path}: {e}", exc_info=True)
        return ""

async def embed_parsed_mineru_data(
    mineru_result: Dict[str, Any],
    original_filename: str,
    milvus_collection_name: str,
    mongo_db_name: str,
    mongo_collection_name: str,
    minio_object_name: Optional[str] = None,
    file_bytes: Optional[bytes] = None  # Add file_bytes to the signature
) -> int:
    """
    Core logic to process MinerU's output, embed it, and store it in databases.
    This function includes an automatic retry mechanism for transient errors.
    """
    max_retries = 2
    retry_delay = 5  # seconds

    for attempt in range(max_retries + 1):
        mongo_client = None
        try:
            mongo_client = get_mongo_client()
            if not mongo_client:
                raise ConnectionError("Failed to connect to MongoDB.")

            if not mineru_result or "result" not in mineru_result:
                logger.error(f"Invalid MinerU data provided for {original_filename}.")
                return 0

            pdf_info = mineru_result.get("result", {})
            
            # --- Enhanced Debug Logging ---
            logger.info(f"Inspecting MinerU result for '{original_filename}':")
            logger.info(f"  - Top-level keys: {list(mineru_result.keys())}")
            if isinstance(pdf_info, dict):
                logger.info(f"  - 'result' object is a DICT with keys: {list(pdf_info.keys())}")
            elif isinstance(pdf_info, list):
                logger.info(f"  - 'result' object is a LIST with {len(pdf_info)} items.")
            else:
                logger.info(f"  - 'result' object is of unexpected type: {type(pdf_info)}")
            # --- End Enhanced Debug Logging ---

            # This part is less likely to fail, so we do it once.
            text_chunks = []
            full_document_text_list = []
            
            # The 'pdf_info' from the new MinerU service is a flat list of content blocks.
            # We can iterate over it directly, removing the old page->blocks nested structure.
            blocks = pdf_info if isinstance(pdf_info, list) else []
            logger.info(f"  - Found {len(blocks)} content blocks to process.")

            # --- ROO: Full MinerU result dump for debugging ---
            try:
                # Use json.dumps for pretty-printing the structure if possible
                logger.info(f"  - ROO_DEBUG: Full MinerU result content:\n{json.dumps(pdf_info, indent=2, ensure_ascii=False)}")
            except Exception as json_e:
                # Fallback for objects that can't be serialized to JSON
                logger.error(f"  - ROO_DEBUG: Could not serialize MinerU result to JSON: {json_e}")
                logger.info(f"  - ROO_DEBUG: Raw MinerU result content: {pdf_info}")
            # --- END ROO DEBUG ---
            for i, block in enumerate(blocks):
                if not isinstance(block, dict):
                    logger.warning(f"  - Block {i} is not a dictionary, it is a {type(block)}. Skipping.")
                    continue

                chunk_content = ""
                block_type = block.get('type')
                if block_type == 'text':
                    chunk_content = block.get('text', '')
                elif block_type == 'table':
                    # Corrected key from 'html_content' to 'table_body' based on logs
                    html_content = block.get('table_body')
                    if html_content:
                        soup = BeautifulSoup(html_content, 'html.parser')
                        chunk_content = soup.get_text(separator=' | ', strip=True)
                # Corrected to handle both 'figure' and 'image' types
                elif block_type in ['figure', 'image']:
                    b64_data = block.get('b64_data')
                    if b64_data:
                        try:
                            image_bytes = base64.b64decode(b64_data)
                            temp_filename = f"figure_{uuid.uuid4()}.png"
                            recognized_text = await _process_image_with_paddle(image_bytes, temp_filename)
                            if recognized_text and recognized_text.strip():
                                chunk_content = f"\n[Image Text: {recognized_text.strip()}]\n"
                            else:
                                latex_code = await call_latexocr_service(image_bytes)
                                if latex_code:
                                    chunk_content = f"\n$${latex_code}$$\n"
                        except Exception as e:
                            logger.error(f"Error processing figure block's b64_data: {e}", exc_info=True)
                    # --- Fallback Logic ---
                    elif file_bytes:
                        page_index = block.get('page_idx')
                        if page_index is not None:
                            logger.warning(f"MinerU identified an image on page {page_index} but provided no data. Attempting fallback to PaddleOCR.")
                            try:
                                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                                    if page_index < len(doc):
                                        page = doc.load_page(page_index)
                                        pix = page.get_pixmap()
                                        img_bytes = pix.tobytes("png")

                                        temp_filename = f"fallback_page_{page_index}.png"
                                        recognized_text = await _process_image_with_paddle(img_bytes, temp_filename)
                                        if recognized_text and recognized_text.strip():
                                            chunk_content = f"\n[Image Text (Fallback OCR): {recognized_text.strip()}]\n"
                                        else:
                                            logger.warning(f"Fallback OCR on page {page_index} yielded no text.")
                                    else:
                                        logger.error(f"Page index {page_index} is out of bounds for document with {len(doc)} pages.")
                            except Exception as e:
                                logger.error(f"Error during PaddleOCR fallback for page {page_index}: {e}", exc_info=True)
                        else:
                            logger.warning(f"MinerU identified an image block but provided no image data (b64_data) or page index. Cannot perform OCR fallback. Block: {block}")
                
                if chunk_content and chunk_content.strip():
                    clean_content = chunk_content.strip()
                    # --- CRITICAL CHANGE: Inject filename into the content for embedding ---
                    # This ensures the vector represents both the filename and the content.
                    content_for_embedding = f"Source Filename: {original_filename}\n\nContent: {clean_content}"
                    text_chunks.append({"content": content_for_embedding, "is_image": "False", "filepath": original_filename})
                    full_document_text_list.append(clean_content) # Keep original clean content for summary
                else:
                    logger.info(f"Skipping empty or invalid block for {original_filename}. Block content: {block}")

            if not text_chunks:
                logger.error(f"MinerU parsing yielded no processable chunks for {original_filename}. Aborting.")
                return 0
            
            full_document_text = "\n\n".join(full_document_text_list)

            db = mongo_client[mongo_db_name]
            documents_collection = db[mongo_collection_name]
            collection = await create_collection(milvus_collection_name)
            
            processed_chunks_milvus = []
            simplified_chunks_mongo = []
            
            for i, chunk in enumerate(text_chunks):
                chunk_id = str(uuid.uuid4())
                # --- DEBUG: Print the exact content being sent to the embedding model ---
                logger.info(f"  - Chunk {i+1} to embed: '{chunk['content'][:500]}...' (length: {len(chunk['content'])})")
                # --- END DEBUG ---
                vector = await get_embedding(chunk["content"]) # This can fail
                if vector is None or vector.size == 0:
                    # Using logger.warning instead of raising an error immediately.
                    # This makes the process more robust to single chunk failures.
                    logger.warning(f"Failed to generate vector for Chunk {i+1}. Skipping this chunk.")
                    continue # Skip to the next chunk
                
                processed_chunks_milvus.append({"id": chunk_id, "vector": vector, "filepath": chunk["filepath"], "content": chunk["content"], "is_image": chunk["is_image"]})
                simplified_chunks_mongo.append({"chunk_id": chunk_id, "content_preview": chunk["content"][:200] + "...", "is_image": False, "page_no": None})

            if not processed_chunks_milvus:
                logger.error(f"No chunks were successfully embedded for {original_filename}. Aborting.")
                return 0

            await insert_to_milvus(collection, processed_chunks_milvus) # This can fail
            
            document_summary = "No text content to summarize."
            if full_document_text:
                document_summary = await summary_documents_content(full_document_text, "Summarize this document.")

            existing_doc = documents_collection.find_one({"original_filename": os.path.basename(original_filename)})
            if existing_doc:
                update_data = {"$set": {"processing_date": datetime.now(), "summary": document_summary, "milvus_chunks": simplified_chunks_mongo, "chunking_strategy": "parser:mineru_splitter:semantic_blocks", "minio_object_path": minio_object_name, "last_embedding_status": "success"}}
                documents_collection.update_one({"_id": existing_doc["_id"]}, update_data)
            else:
                document_data = {"original_filename": os.path.basename(original_filename), "processing_date": datetime.now(), "summary": document_summary, "milvus_chunks": simplified_chunks_mongo, "chunking_strategy": "parser:mineru_splitter:semantic_blocks", "minio_object_path": minio_object_name, "last_embedding_status": "success"}
                documents_collection.insert_one(document_data)

            logger.info(f"Successfully processed and embedded {len(processed_chunks_milvus)} chunks for {original_filename}.")
            return len(processed_chunks_milvus)

        except (ConnectionError, ValueError, Exception) as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries + 1} failed for '{original_filename}'. Error: {e}", exc_info=True)
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.critical(f"All retry attempts failed for '{original_filename}'.")
                raise e  # Re-raise the exception after all retries have failed
        finally:
            if mongo_client:
                mongo_client.close()
    return 0 # Should not be reached if successful


async def process_and_embed_pdf(
    file_bytes: bytes,
    original_filename: str,
    milvus_collection_name: str,
    mongo_db_name: str,
    rag_id: int,
    mongo_collection_name: str = "documents"
):
    """
    Processes a PDF's byte content using the configured MinerU processor (local or remote),
    and then calls the core embedding logic.
    """
    logger.info(f"Starting PDF processing for: {original_filename} using configured MinerU processor.")

    async with doc_processing_lock:
        logger.info(f"Acquired lock for processing {original_filename}")
        try:
            # 1. Connect to Milvus (lightweight check)
            await connect_to_milvus()
            
            # 2. Process with the unified MinerU processor
            # The strategy is now handled within process_pdf_with_mineru
            logger.info(f"Step 2 - Calling unified MinerU processor (backend: '{settings.MINERU_PARSE_BACKEND}')")
            
            mineru_result = await process_pdf_with_mineru(file_bytes, original_filename)
            
            if not mineru_result or "result" not in mineru_result:
                logger.error(f"MinerU processing failed to return valid data for {original_filename}. Aborting.")
                # Consider updating status to failed here
                return 0

            mineru_chunks = mineru_result["result"]
            logger.info(f"Step 2 - MinerU parsing complete. Found {len(mineru_chunks)} chunks.")

            if not mineru_chunks:
                logger.warning(f"MinerU parsing yielded no processable chunks for {original_filename}. Aborting.")
                return 0

            # 4. Call the restored and refactored embedding logic
            logger.info("Step 3 - Calling core embedding logic...")
            
            # We don't need to store the MinerU result separately in MinIO for this flow,
            # as we are processing it immediately. The `minio_object_name` can be None.
            count = await embed_parsed_mineru_data(
                mineru_result=mineru_result,
                original_filename=original_filename,
                milvus_collection_name=milvus_collection_name,
                mongo_db_name=mongo_db_name,
                mongo_collection_name=mongo_collection_name,
                minio_object_name=None, # This flow processes immediately, no need to save intermediate JSON
                file_bytes=file_bytes # Pass the original file bytes for the fallback logic
            )
            
            return count

        except Exception as e:
            logger.critical(f"An error occurred during the PDF processing for '{original_filename}': {e}", exc_info=True)
            # Optionally, update MongoDB to reflect the failure
            mongo_client = get_mongo_client()
            if mongo_client:
                try:
                    db = mongo_client[mongo_db_name]
                    documents_collection = db[mongo_collection_name]
                    documents_collection.update_one(
                        {"original_filename": os.path.basename(original_filename)},
                        {"$set": {"last_embedding_status": f"failed: {e}"}},
                        upsert=True
                    )
                finally:
                    mongo_client.close()
            return 0
        finally:
            logger.info(f"Released lock for {original_filename}")


async def process_markdown_file(file_path: str, image_dir: str, collection_name: str, mongo_db_name: str, mongo_collection_name: str = "documents"):
    """Hinweis"""
    # Connect to Milvus
    await connect_to_milvus()

    # Connect to MongoDB
    mongo_client = get_mongo_client()
    if not mongo_client:
        logger.error("Failed to connect to MongoDB. Aborting process.")
        return 0

    db = mongo_client[mongo_db_name]
    documents_collection = db[mongo_collection_name]

    # Extract original filename and paths to generated files (assuming standard output structure from pdf.py)
    original_filename = os.path.basename(file_path).replace(".md", "")
    processed_img_dir = os.path.join(os.path.dirname(file_path), "images") # Assuming images are in a subdir named 'images'
    processed_layout_pdf_path = os.path.join(PDF_PATH, original_filename.join("_layout.pdf"))
    processed_model_pdf_path = os.path.join(PDF_PATH, original_filename.join("_model.pdf"))

    # Create collection in Milvus
    collection = await create_collection(collection_name)

    # Read Markdown file
    content = await read_markdown_file(file_path)

    # Parse Markdown
    # Pass the image_dir to parse_markdown
    chunks = await parse_markdown(content, file_path, image_dir)

    # Process chunks (get embeddings and prepare for Milvus and MongoDB)
    processed_chunks_milvus = []
    simplified_chunks_mongo = []
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        vector = await get_embedding(chunk["content"])
        if vector is None or vector.size == 0:
            logger.warning(f"Failed to generate vector for markdown chunk. Skipping. Content: {chunk['content'][:100]}...")
            continue

        processed_chunks_milvus.append({
            "id": chunk_id,
            "vector": vector,
            "filepath": chunk["filepath"],
            "content": chunk["content"],
            "is_image": chunk["is_image"]
        })

        # Create simplified chunk representation for MongoDB
        simplified_chunks_mongo.append({
            "chunk_id": chunk_id,
            "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
            "is_image": chunk["is_image"] == "True",
            # TODO: Add page number if available from the parsing process
        })


    # Insert into Milvus
    await insert_to_milvus(collection, processed_chunks_milvus)

    # Generate Summary
    # Concatenate text chunks for summarization
    text_content = " ".join([chunk["content"] for chunk in chunks if chunk["is_image"] == "False"])
    if text_content:
        document_summary = await summary_documents_content(text_content, "Summarize this document.") # Using a generic question for summarization
    else:
        document_summary = "No text content to summarize."

    # Prepare document data for MongoDB
    document_data = {
        "original_filename": original_filename,
        "processed_md_path": file_path,
        "processed_img_dir": processed_img_dir,
        "processed_layout_pdf_path": processed_layout_pdf_path,
        "processed_model_pdf_path": processed_model_pdf_path,
        "processing_date": datetime.now(),
        "summary": document_summary,
        "milvus_chunks": simplified_chunks_mongo
    }

    # Insert into MongoDB
    try:
        insert_result = documents_collection.insert_one(document_data)
        logger.info(f"Inserted document metadata into MongoDB with ID: {insert_result.inserted_id}")
    except Exception as e:
        logger.error(f"Failed to insert document metadata into MongoDB: {e}", exc_info=True)
        # Consider logging this error and potentially cleaning up Milvus entries if MongoDB insertion fails
    finally:
        # Close MongoDB connection
        if mongo_client:
            mongo_client.close()

    return len(processed_chunks_milvus)


def get_file_gist_by_filename_and_rag_id(db: Session, filename: str, rag_id: int) -> Optional[schemas.FileGist]:
    """
    Retrieves a FileGist by its original filename and associated RAG Data ID,
    and generates a pre-signed MinIO download URL.
    (Moved here from rag_file_service to resolve circular import)
    """
    # Query for the file using both filename and rag_data_id for specificity
    all_gists_for_rag = db.query(FileGist).filter(FileGist.rag_id == rag_id).all()

    # Then, find the one that ends with the desired filename in Python
    file_gist = None
    for gist in all_gists_for_rag:
        if gist.filename.endswith(filename):
            file_gist = gist
            break

    if not file_gist:
        return None

    minio_bucket_name = settings.MINIO_BUCKET_NAME
    if not minio_bucket_name:
        print(f"MINIO_BUCKET_NAME not set. Cannot generate download URL for {filename}.")
        return schemas.FileGist.from_orm(file_gist)

    try:
        minio_client = get_minio_client()
        download_url = minio_client.presigned_get_object(
            minio_bucket_name,
            file_gist.filename,
            expires=timedelta(days=7),
        )
        file_gist_schema = schemas.FileGist.from_orm(file_gist)
        file_gist_schema.download_url = download_url
        return file_gist_schema
    except S3Error as e:
        print(f"MinIO Error generating pre-signed URL for file {filename}: {e}")
        return schemas.FileGist.from_orm(file_gist)
    except Exception as e:
        print(f"An unexpected error occurred during MinIO URL generation for file {filename}: {e}")
        return schemas.FileGist.from_orm(file_gist)

# MongoDB Connection
def get_mongo_client():
    """Establishes and returns a MongoDB client connection."""
    try:
        # Use the ServerApi version 1
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        # Ping the admin database to confirm a successful connection
        client.admin.command('ping')
        logger.info("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        return None


# Kommentar
# async def main():
#     pass
    # # Kommentar
    # # Kommentar
    # pp_folder = "/Users/liuzhengyin/Downloads/GB3000"

    # logger.info(f"Kommentar{pp_folder} Kommentar")
    # files = read_all_files(pp_folder)

    # logger.info(f"Kommentar{len(files)} Kommentar")
    # logger.info("Kommentar", files)
    # for i in files:
    #     if i.endswith(".pdf"):
    #         deal_to_md(i)

    # markdown_file = "/Users/liuzhengyin/Documents/AI/websev/chr/lib/deal_documents/output/IEC+60601-1+2012A1.md"
    # markdown_file = "/Users/Jinglu/Downloads/md, SP04_P001_Nonconforming control procedureKommentar"
    # collection_name = "markdown_data"
    #
    # Kommentar
    # files = read_all_files(md_path)
    # for i in files:
        #  if i.endswith(".pdf"):
    # count = await process_markdown_file(markdown_file, collection_name)
    # logger.info(f"Kommentar{count} Kommentar")

    # # Kommentar
    # query = (
    #     "If however there is some insulation at both points A and B, then no part of the "
    #     "SECONDARY CIRCUIT is 'live' according to the definition in the second edition, so "
    #     "the second edition of this standard specifies no requirements for that insulation, "
    #     "which can therefore discovered this problem in 1993, unfortunately just too late "
    #     "for it to be dealt with in the second (and final) amendment to the second edition "
    #     "of this standard. The approach adopted in this edition is intended to overcome this problem."
    # )
    # await connect_to_milvus()
    # collection = Collection(collection_name)
    # results = await search_in_milvus(collection, query)
    # logger.info("\nAbfragenErgebnisse:")
    # for i, result in enumerate(results):
    #     logger.info(f"\nErgebnisse {i + 1}:")
    #     logger.info(f"Kommentar{1 - result['distance']:.4f}")
    #     logger.info(f"Typ: {'Kommentar' if result['is_image'] == 'True' else 'Kommentar'}")
    #     logger.info(f"Kommentar{result['filepath']}")
    #     logger.info(f"Kommentar{result['content'][:100]}..." if len(result['content']) > 100 else result['content'])


# if __name__ == "__main__":
#     asyncio.run(main())
