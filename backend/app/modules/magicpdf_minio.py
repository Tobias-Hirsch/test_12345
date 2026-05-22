import asyncio
import logging
from app.services.mineru_service import mineru_client

# --- Kommentar
# Kommentar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """
    Hinweis
    """
    logger.info("--- Hinweis")

    # 1. Kommentar
    #    Kommentar"bucket_name/path/to/your/file.pdf"
    #    Kommentar
    document_to_process = "your-bucket-name/your-document-path.pdf"
    logger.info(f"Hinweis{document_to_process}")

    # 2. Kommentar
    #    Kommentar
    result = await mineru_client.process_document(document_path=document_to_process)

    # 3. Kommentar
    if result:
        logger.info("Hinweis")
        # Kommentar
        # Kommentar'data'. 
        processed_data = result.get("data", {})
        content_preview = str(processed_data)[:500] # Hinweis
        logger.info(f"Hinweis{content_preview}...")
    else:
        logger.error("Fehler bei der Verarbeitung")

    logger.info("--- Hinweis")


if __name__ == "__main__":
    # Kommentar
    # Kommentar
    # python -m app.modules.magicpdf_minio
    asyncio.run(main())
