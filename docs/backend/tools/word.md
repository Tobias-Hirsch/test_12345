# `tools/word.py` - Word DokumenteKommentar

Hinweis`backend/app/tools/word.py` Hinweis

## Kommentar
*   **Word Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`python-docx` Hinweis

Hinweis
```python
from docx import Document
import logging

logger = logging.getLogger(__name__)

def deal_to_text(docx_path: str) -> str:
    """
    Extracts all text from a Word (.docx) file.
    """
    extracted_text = []
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            extracted_text.append(paragraph.text)
        logger.info(f"Successfully extracted text from Word: {docx_path}")
    except Exception as e:
        logger.error(f"Error processing Word file {docx_path}: {e}")
        raise
    return "\n\n".join(extracted_text)

# Kommentar
# def extract_tables_from_docx(docx_path: str) -> List[List[List[str]]]:
#     document = Document(docx_path)
#     tables_data = []
#     for table in document.tables:
#         current_table_data = []
#         for row in table.rows:
#             row_data = []
#             for cell in row.cells:
#                 row_data.append(cell.text)
#             current_table_data.append(row_data)
#         tables_data.append(current_table_data)
#     return tables_data
```

## Kommentar
`/backend/app/tools/word.py`