# `tools/pdf.py` - PDF Kommentar

Hinweis`backend/app/tools/pdf.py` Hinweis

## Kommentar
*   **PDF Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`PyPDF2` Oder `pymupdf` (fitz) Hinweis

Hinweis`PyPDF2` Hinweis
```python
from PyPDF2 import PdfReader
import logging

logger = logging.getLogger(__name__)

def deal_to_md(pdf_path: str) -> str:
    """
    Extracts text from a PDF file and converts it into a simple Markdown format.
    """
    markdown_content = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text()
                if text:
                    markdown_content.append(f"## Page {i + 1}\n\n")
                    markdown_content.append(text.strip())
                    markdown_content.append("\n\n---\n\n") # Hinweis
            logger.info(f"Successfully processed PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing PDF file {pdf_path}: {e}")
        raise
    return "\n".join(markdown_content)

# Kommentar
# import fitz # PyMuPDF

# def deal_to_md_pymupdf(pdf_path: str) -> str:
#     markdown_content = []
#     try:
#         doc = fitz.open(pdf_path)
#         for page_num in range(doc.page_count):
#             page = doc.load_page(page_num)
#             text = page.get_text("text") # "text" for plain text, "html", "json", "xml"
#             if text:
#                 markdown_content.append(f"## Page {page_num + 1}\n\n")
#                 markdown_content.append(text.strip())
#                 markdown_content.append("\n\n---\n\n")
#         doc.close()
#         logger.info(f"Successfully processed PDF with PyMuPDF: {pdf_path}")
#     except Exception as e:
#         logger.error(f"Error processing PDF file {pdf_path} with PyMuPDF: {e}")
#         raise
#     return "\n".join(markdown_content)
```

## Kommentar
`/backend/app/tools/pdf.py`