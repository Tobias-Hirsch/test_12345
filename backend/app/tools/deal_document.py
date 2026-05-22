# Developer: Jinglu Han
# mailbox: admin@de-manufacturing.cn

import re
import json
import logging
import logging
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from fastapi import UploadFile
from app.llm.llm import get_llm
from app.tools.pdf import process_pdf_with_mineru
from app.tools.word import extract_word_content_from_url, extract_word_content_from_bytes
from app.tools.exlsx import download_and_parse_xlsx, extract_excel_content_from_bytes
from app.tools.split_tools import semantic_text_splitter
from app.llm.llm import llm_qwen_vl_max_ainvoke, llm_ollama_vision_ainvoke
import asyncio
import os
import json
from urllib.parse import urlparse
import httpx
from ..core.config import settings # Global import

logger = logging.getLogger(__name__)

DOCUMENT_TYPE_QWEN_VL_DEAL_IMAGE = settings.DOCUMENT_TYPE_QWEN_VL_DEAL_IMAGE

class SummarizeQuestionContentType(BaseModel):
    summarize: str = Field(description="A concise summary of the provided document text.")


SUMMARY_DOCUMENTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You summarize industrial and technical documents for retrieval systems.
Write a concise, factual summary grounded only in the provided document text.
Do not mention parser errors, formatting instructions, or implementation details.
If the document text is noisy or partially extracted, keep only the reliable content.
""".strip(),
        ),
        (
            "human",
            """
Question:
{question}

Document text:
{document_text}
""".strip(),
        ),
    ]
)

async def process_image(doc_info,question):
    url = doc_info['url']
    file_extension = os.path.splitext(urlparse(url).path)[-1].lower()
    try:
        if file_extension in DOCUMENT_TYPE_QWEN_VL_DEAL_IMAGE:
            content = await llm_qwen_vl_max_ainvoke(question,url)
            return content
        return None
    except Exception as e:
        return f"Fehler bei der Bildverarbeitung: {str(e)}"

async def process_document(doc_info):
    """Hinweis"""  
    url = doc_info['url']
    file_extension = os.path.splitext(urlparse(url).path)[-1].lower()
    try:
        if file_extension in ['.xlsx', '.xls']:
            content = await download_and_parse_xlsx(url)
        elif file_extension in ['.pdf']:
            # To call the new function, we must first download the content from the URL.
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()  # Raise an exception for bad status codes
                file_bytes = await response.aread()
            
            filename = os.path.basename(urlparse(url).path)
            mineru_result = await process_pdf_with_mineru(file_bytes, filename)
            
            # The result from mineru is a dictionary, e.g., {"result": [...]}.
            # We need to decide how to represent this as 'content'.
            # For now, let's serialize the whole result to JSON string.
            if mineru_result:
                content = json.dumps(mineru_result, ensure_ascii=False)
            else:
                content = ""

        elif file_extension in ['.doc', '.docx']:
            content = await extract_word_content_from_url(url)
        else:
            return None

        return content
    except Exception as e:
        return f"Hinweis{str(e)}"


async def process_images(docu_list, question, max_concurrency=5):
    """Hinweis

    Args:
        docu_list: DokumenteHinweis
        question: BenutzerHinweis
        max_concurrency: Hinweis

    Returns:
        Hinweis
    """
    # Kommentar
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_image_with_semaphore(doc):
        async with semaphore:
            return await process_image(doc, question)

    # Kommentar
    tasks = [process_image_with_semaphore(doc) for doc in docu_list]

    # Kommentarührt ausKommentar
    results = await asyncio.gather(*tasks)
    # Kommentar
    all_contents = []
    for content in results:
        if isinstance(content, str) and content:
            all_contents.append(content)

    return all_contents

async def process_documents(docu_list):
    """Hinweis"""
    # Kommentar
    tasks = [process_document(doc) for doc in docu_list]

    # Kommentarührt ausKommentar
    results = await asyncio.gather(*tasks)

    # Kommentar
    all_contents = []
    for content in results:
        if isinstance(content, str) and content:
            # If content is a JSON string from a PDF, we need to handle it differently.
            # This part of the logic might need further review based on how this function is used.
            # For now, we attempt to split it as text.
            chunks = await semantic_text_splitter(content)
            if chunks is not None:
                all_contents.append(chunks)

    return all_contents


async def summary_documents_content(docu_str: str, question: str):
    document_text = str(docu_str).strip()
    if not document_text:
        return None

    llm_instance = get_llm(show_think_process=False)

    try:
        chain = SUMMARY_DOCUMENTS_PROMPT | llm_instance.with_structured_output(
            SummarizeQuestionContentType
        ).with_retry(stop_after_attempt=3)
        result = await chain.ainvoke(
            {"document_text": document_text, "question": str(question)},
            config={"run_name": f"summarize_question_content_{str(question)}"}
        )
        summary = getattr(result, "summarize", None)
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    except Exception as e:
        logger.warning(f"Structured document summarization failed: {e}")

    fallback_prompt = (
        "Summarize the following document text in 2-4 factual sentences. "
        "Ignore extraction noise and do not mention formatting or parser issues.\n\n"
        f"Question: {question}\n\n"
        f"Document text:\n{document_text}"
    )

    try:
        fallback_result = await llm_instance.ainvoke(fallback_prompt)
        fallback_text = fallback_result.content if hasattr(fallback_result, "content") else str(fallback_result)
        fallback_text = fallback_text.strip()
        return fallback_text or None
    except Exception as e:
        logger.error(f"Fallback document summarization failed: {e}")
        return None


async def process_and_summarize_documents(doc_list, question, max_concurrency=5, max_length=2000):
    """
    Hinweis

    Args:
        doc_list: DokumenteHinweis
        question: BenutzerHinweis
        max_concurrency: Hinweis
        max_length: Hinweis

    Returns:
        Hinweis
    """
    # Kommentar
    chunks = await process_documents(doc_list)
    if not chunks:
        return []

    # Kommentar
    semaphore = asyncio.Semaphore(max_concurrency)

    async def summarize_with_semaphore(chunk):
        async with semaphore:
            return await summary_documents_content(chunk, question)

    # Kommentar
    tasks = [summarize_with_semaphore(chunk) for chunk in chunks]
    summaries = await asyncio.gather(*tasks)

    # Kommentar
    summaries = [s for s in summaries if s and s.strip()]

    # Kommentar
    while summaries and sum(len(s) for s in summaries) > max_length:
        # Kommentar
        batch_size = max(1, len(summaries) // 2)
        new_summaries = []

        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i + batch_size]
            combined_text = "\n".join(batch)
            # Kommentar
            summary = await summary_documents_content(combined_text, question)
            if summary and summary.strip():
                new_summaries.append(summary)

        summaries = new_summaries

    return summaries


async def process_documents_and_images(doc_list, question, max_concurrency=5, max_length=2000):
    """
    Hinweis

    Args:
        doc_list: DokumenteHinweis
        question: BenutzerHinweis
        max_concurrency: Hinweis
        max_length: Hinweis

    Returns:
        Hinweis
    """
    # Kommentar
    doc_task = process_and_summarize_documents(doc_list, question, max_concurrency, max_length)
    image_task = process_images(doc_list, question, max_concurrency)  # Hinweis

    # Kommentar
    doc_results, image_results = await asyncio.gather(doc_task, image_task)

    # Kommentar
    combined_results = []
    combined_results.extend(doc_results)
    combined_results.extend(image_results)

    return "\n".join(combined_results)


def _extract_text_from_mineru_result(result: dict) -> str:
    """
    Hinweis
    Hinweis{"result": [{"type": "text", "text": "...", "content": "..."}, ...]}
    """
    if not result or "result" not in result or not isinstance(result["result"], list):
        return ""

    text_parts = []
    for item in result["result"]:
        # --- DEBUG: Print the full structure of each item ---
        logger.info(f"--- MinerU Result Item: {item} ---")
        # --- END DEBUG ---
        # Broaden the extraction to include any block that has a 'content' key.
        # This will capture text, titles, headers, list items, etc.
        if isinstance(item, dict):
            if "text" in item:
                text_parts.append(str(item["text"]))
            elif "content" in item:
                text_parts.append(str(item["content"]))
    
    extracted_text = "\n".join(text_parts)
    logger.info(f"--- EXTRACTED TEXT from MinerU result (Length: {len(extracted_text)}) ---")
    return extracted_text


def _extract_text_with_pymupdf_fallback(file_bytes: bytes) -> str:
    """
    A fallback function to extract raw text using PyMuPDF if MinerU fails.
    """
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        logger.info(f"--- PyMuPDF Fallback: Successfully extracted text (Length: {len(text)}) ---")
        return text
    except Exception as e:
        logger.error(f"--- PyMuPDF Fallback failed: {e} ---")
        return ""


async def extract_text_from_file_content(file_content: bytes, filename: str, question: str = None) -> str:
    """
    Hinweis
    
    Args:
        file_content: Hinweis
        filename: Dateiname. 
        question: BenutzerHinweis

    Returns:
        Hinweis
    """
    file_extension = os.path.splitext(filename)[-1].lower()

    # Kommentar
    # DOCUMENT_TYPE_QWEN_VL_DEAL_IMAGE JaKommentar
    supported_image_types = DOCUMENT_TYPE_QWEN_VL_DEAL_IMAGE
    if file_extension.strip('.') in supported_image_types:
        if not question:
            return "Cannot analyze image without a question."
        return await llm_ollama_vision_ainvoke(question, file_content)

    if file_extension == '.pdf':
        # Kommentar
        mineru_result = await process_pdf_with_mineru(file_content, filename)
        
        # --- DEBUG: Log the raw MinerU result to see its structure and type ---
        logger.info("--- RAW MINERU RESULT ---")
        logger.info(f"--- Type: {type(mineru_result)} ---")
        logger.info(mineru_result)
        logger.info("--- END RAW MINERU RESULT ---")
        # --- END DEBUG ---

        mineru_result_dict = None

        if isinstance(mineru_result, dict):
            # Case 1: The result is already a dictionary
            mineru_result_dict = mineru_result
        elif isinstance(mineru_result, str):
            # Case 2: The result is a JSON string
            try:
                mineru_result_dict = json.loads(mineru_result)
            except json.JSONDecodeError:
                # logger.error("Failed to decode MinerU JSON result string.")
                return ""
        else:
            # Case 3: The result is neither a dict nor a string (e.g., None)
            return ""

        # Now, process the dictionary if we have one
        if mineru_result_dict:
            return _extract_text_from_mineru_result(mineru_result_dict)
        else:
            return ""
    
    elif file_extension in ['.doc', '.docx']:
        # Kommentar
        logger.info(f"Hinweis{filename}")
        result = await extract_word_content_from_bytes(file_content)
        
        if result and not result.startswith("Fehler") and not result.startswith("Hinweis"):
            logger.info(f"WordDokument erfolgreich verarbeitet: {filename}")
        else:
            logger.error(f"WordDokumenteVerarbeitung fehlgeschlagen: {filename}, Ergebnisse: {result[:100] if result else 'None'}...")
        
        return result
    elif file_extension in ['.xlsx', '.xls']:
        # Kommentar
        return await extract_excel_content_from_bytes(file_content, filename)
    else:
        return f"Unsupported file type for preview: {file_extension}"

async def get_text_from_uploaded_file(file: UploadFile) -> str:
    """
    Hinweis
    Hinweis

    Args:
        file: FastAPI Hinweis

    Returns:
        Hinweis
    """
    filename = file.filename
    file_content = await file.read()
    return await extract_text_from_file_content(file_content, filename)

# async def main():
#     doc = [
#         {
#             "file_type": "dsad ipsum",
#             "file_name": "Excepteasdasse ipsum",
#             "url": "http://localhost:8080/image/rosti_ai_logo.png"
#         },
#         {
#             "file_type": "Excepteur dolore in esse ipsum",
#             "file_name": "Excepteur dolore in esse ipsum",
#             "url": "http://localhost:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL21tLXJhZy1idWNrZXQvcm9zdGkvQ09QMDNfRjAxNi0wMF9Nb3VsZGluZyUyMGZpcnN0JTIwb3IlMjBsYXN0JTIwc2FtcGxlJTIwaW5zcGVjdGlvbiUyMHJlY29yZCVFNiVCMyVBOCVFNSVBMSU5MSVFOSVBNiU5NiVFNiU5QyVBQiVFNiVBMCVCNyVFNiVBMyU4MCVFOSVBQSU4QyVFOCVBRSVCMCVFNSVCRCU5NV9BMC54bHM_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1MWlJQWDJVWFY1U1lTUE01QzhFRCUyRjIwMjUwNTI3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUyN1QwODM5MjRaJlgtQW16LUV4cGlyZXM9NDMxOTgmWC1BbXotU2VjdXJpdHktVG9rZW49ZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmhZMk5sYzNOTFpYa2lPaUpNV2xKUVdESlZXRlkxVTFsVFVFMDFRemhGUkNJc0ltVjRjQ0k2TVRjME9ETTNPRE0wTkN3aWNHRnlaVzUwSWpvaWJXbHVhVzloWkcxcGJpSjkuR1c5UzItazNqMW1TLVVzVzBpS2tBSTNJUWV2STBRY280VzlzMV9pR3Z5dGUxazdhS2F0TUMyNzhjcVkwT1JBS1Z0OFJSY3pBZ2t4Tk9TWm90aG9NV2cmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnZlcnNpb25JZD1udWxsJlgtQW16LVNpZ25hdHVyZT0wNDFhNzRlMDVlZTAzYjJjNjYxMjBhYWViZmQwNDU0ODBkYzAwYTdjMDBkNzJhZTUyY2Q5MjAwNWU3MGExNzhj"
#         }
#     ]
#     #a = await process_documents_and_images(doc, "Kommentar")
#     a = await process_and_summarize_documents(doc, "Kommentar")
#     print(a)
#     print(len(a))


# if __name__ == "__main__":
#     asyncio.run(main())
