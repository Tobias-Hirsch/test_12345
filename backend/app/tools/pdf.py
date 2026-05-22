import os
import logging
from typing import Tuple, Optional
import fitz  # PyMuPDF
from ..core.config import settings
from app.services.mineru_unified_service import get_unified_mineru_processor

# --- Kommentar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Kommentar
async def process_pdf_with_mineru(file_bytes: bytes, filename: str) -> Optional[dict]:
    """
    Hinweis
    Hinweis

    :param file_bytes: PDF Hinweis
    :param filename: Hinweis
    :return: MinerU Hinweis
    """
    strategy = settings.PDF_PROCESSING_STRATEGY
    logger.info(f"Hinweis{filename}, Hinweis{strategy}")
    
    # Kommentar
    processor = get_unified_mineru_processor()
    
    # Kommentar
    result = await processor.process_document_bytes(file_bytes, filename, strategy=strategy)
    
    if not result:
        logger.error(f"Fehler bei der Verarbeitung{filename} (Fehler bei der Verarbeitung{strategy})")
        return None
    return result


# --- Kommentar

def read_all_files(folder_path: str) -> list:
    """Hinweis"""
    files_list = []
    if not os.path.exists(folder_path):
        logger.warning(f"Warnhinweis{folder_path} Warnhinweis")
        return files_list
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files_list.append(filename)
    return files_list


def fix_pdf(file_path: str) -> Optional[str]:
    """
    Hinweis

    :param file_path: PDF Hinweis
    :return: Hinweis
    """
    doc = None  # Hinweis
    try:
        doc = fitz.open(file_path)
        fixed_file_path = file_path.replace(".pdf", ".fixed.pdf")
        doc.save(fixed_file_path)
        logger.info(f"PDF Hinweis{file_path} -> {fixed_file_path}")
        return fixed_file_path
    except Exception as e:
        logger.error(f"PDF Fehler bei der Verarbeitung{file_path}: {e}")
        return None
    finally:
        if doc:
            doc.close()


def check_pdf_integrity(file_path: str) -> bool:
    """Hinweis"""
    try:
        doc = fitz.open(file_path)
        # Kommentar
        is_ok = doc.is_pdf and doc.page_count > 0
        doc.close()
        return is_ok
    except Exception as e:
        logger.error(f"PDF Fehler bei der Verarbeitung{file_path}: {e}")
        return False


def extract_raw_text_with_pymupdf(file_path: str) -> Tuple[str, int]:
    """
    Hinweis

    :param file_path: PDF Hinweis
    :return: Hinweis
    """
    try:
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        page_count = doc.page_count
        doc.close()
        logger.info(f"Hinweis{file_path} Hinweis{page_count}")
        return text, page_count
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung{file_path} Fehler bei der Verarbeitung{e}")
        return "", 0

