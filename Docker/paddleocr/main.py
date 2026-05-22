import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import logging
import tempfile
import os
from pathlib import Path
import io

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PaddleOCR Service",
    description="Exposes PaddleOCR functionality over an HTTP API for plain text extraction.",
    version="2.0.0",
)

# --- Model Loading ---
try:
    logger.info("Initializing PaddleOCR...")
    # Initialisiert das OCR-Modell für deutsche bzw. lateinische Texte.
    # Die Sprache kann über PADDLEOCR_LANG überschrieben werden, Standard ist Deutsch.
    # use_angle_cls=True hilft bei der Korrektur der Textausrichtung.
    paddleocr_lang = os.getenv('PADDLEOCR_LANG', 'german')
    logger.info(f"Using PaddleOCR language: {paddleocr_lang}")
    ocr_model = PaddleOCR(use_angle_cls=True, lang=paddleocr_lang)
    logger.info("PaddleOCR model initialized successfully.")
except Exception as e:
    logger.error("!!! CRITICAL: An exception occurred during PaddleOCR initialization !!!")
    logger.error(f"Exception type: {type(e).__name__}")
    logger.error(f"Exception details: {e}", exc_info=True)
    ocr_model = None

@app.get("/health", summary="Health Check")
def health_check():
    """
    Simple health check endpoint to verify the service is running and the model is loaded.
    """
    if ocr_model is None:
        raise HTTPException(status_code=500, detail="OCR model could not be loaded.")
    return {"status": "ok", "model_ready": True}

@app.post("/ocr", summary="Perform OCR on a single image")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Receives an image file, performs OCR, and returns the extracted plain text.
    This endpoint is designed for speed and low memory usage.
    It does not perform layout analysis.
    """
    if ocr_model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: OCR model is not loaded.")

    logger.info(f"Received file: {file.filename}, content-type: {file.content_type}")

    # Read file into memory
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Could not read uploaded file.")

    # Perform OCR using the lightweight model
    try:
        logger.info(f"Performing OCR prediction on {file.filename}...")
        # The ocr method takes image bytes directly
        # The `use_angle_cls=True` in the constructor is sufficient.
        # The `ocr` method itself does not take a `cls` argument.
        result = ocr_model.ocr(contents)
        logger.info("OCR prediction completed.")

        # The result is a list of lists of lines.
        # e.g., [[[[box], ('text', score)], [[box], ('text', score)]]]
        # We need to extract and join all text parts.
        lines = result[0] if result else []
        text_lines = [line[1][0] for line in lines]
        full_text = "\n".join(text_lines)
        
        logger.info("Successfully extracted text from image.")
        
        return {
            "filename": file.filename,
            "text": full_text,
        }

    except Exception as e:
        logger.error(f"An error occurred during OCR processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during OCR processing.")

if __name__ == "__main__":
    import uvicorn
    # This block is for local debugging and won't be used in the Docker container
    uvicorn.run(app, host="0.0.0.0", port=8080)