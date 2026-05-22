# `tools/exlsx.py` - Excel Kommentar

Hinweis`backend/app/tools/exlsx.py` Hinweis

## Kommentar
*   **Excel Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`openpyxl` Oder `pandas` Hinweis

Hinweis
```python
import openpyxl
import logging

logger = logging.getLogger(__name__)

def deal_to_text(xlsx_path: str) -> str:
    """
    Extracts text content from an Excel (.xlsx) file.
    Combines text from all sheets and cells.
    """
    extracted_text = []
    try:
        workbook = openpyxl.load_workbook(xlsx_path)
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            extracted_text.append(f"## Sheet: {sheet_name}\n\n")
            for row in sheet.iter_rows():
                row_values = []
                for cell in row:
                    if cell.value is not None:
                        row_values.append(str(cell.value))
                if row_values:
                    extracted_text.append(" ".join(row_values))
            extracted_text.append("\n\n---\n\n") # Hinweis
        logger.info(f"Successfully processed Excel: {xlsx_path}")
    except Exception as e:
        logger.error(f"Error processing Excel file {xlsx_path}: {e}")
        raise
    return "\n".join(extracted_text)
```

## Kommentar
`/backend/app/tools/exlsx.py`