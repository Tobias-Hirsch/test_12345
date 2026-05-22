# `tools/deal_document.py` - DokumenteKommentar

Hinweis`backend/app/tools/deal_document.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar`.docx`, `.xlsx`, `.pdf`)Kommentar
*   **Kommentar**: Kommentar
*   **FehlerKommentar**: Kommentar

## Kommentar
Hinweis`pdf.py`, `word.py`, `exlsx.py`)Hinweis

Hinweis
```python
import os
from backend.app.tools.pdf import deal_to_md as process_pdf_to_md
# from backend.app.tools.word import deal_to_md as process_word_to_md # Kommentar
# from backend.app.tools.exlsx import deal_to_text as process_excel_to_text # Kommentar
import logging

logger = logging.getLogger(__name__)

async def process_document_to_text(file_path: str) -> str:
    """
    Processes a document based on its file type and returns its content as text.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        return process_pdf_to_md(file_path)
    # elif file_extension == ".docx":
    #     return process_word_to_md(file_path)
    # elif file_extension == ".xlsx":
    #     return process_excel_to_text(file_path)
    else:
        logger.warning(f"Unsupported file type for processing: {file_extension}")
        raise ValueError(f"Unsupported file type: {file_extension}")

```

## Kommentar
`/backend/app/tools/deal_document.py`