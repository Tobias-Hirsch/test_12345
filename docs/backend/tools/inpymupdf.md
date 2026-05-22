# `tools/inpymupdf.py` - PyMuPDF(Fitz)Kommentar

Hinweis`backend/app/tools/inpymupdf.py` Hinweis

## Kommentar
*   **PDF Kommentar**: HochKommentar
*   **PDF Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **PDF Kommentar**: Kommentar

## Kommentar
Hinweis`fitz`(PyMuPDF Hinweis

Hinweis
```python
import fitz # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf_pymupdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF (fitz).
    """
    text_content = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text") # "text" for plain text
            if text:
                text_content.append(text.strip())
        doc.close()
        logger.info(f"Successfully extracted text from PDF with PyMuPDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF file {pdf_path} with PyMuPDF: {e}")
        raise
    return "\n\n".join(text_content)

# Kommentar
# def extract_images_from_pdf(pdf_path: str, output_dir: str):
#     doc = fitz.open(pdf_path)
#     for i in range(len(doc)):
#         for img in doc.get_page_images(i):
#             xref = img[0]
#             pix = fitz.Pixmap(doc, xref)
#             if pix.n - pix.alpha < 4:  # this is GRAY or RGB
#                 pix.save(os.path.join(output_dir, f"page{i}-{xref}.png"))
#             else:  # CMYK: convert to RGB first
#                 pix.set_alpha(pix.alpha)
#                 pix.save(os.path.join(output_dir, f"page{i}-{xref}.png"))
#             pix = None
```

## Kommentar
`/backend/app/tools/inpymupdf.py`