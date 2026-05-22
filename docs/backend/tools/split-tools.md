# `tools/split_tools.py` - Kommentar

Hinweis`backend/app/tools/split_tools.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
Hinweis`RecursiveCharacterTextSplitter` OderHinweis

Hinweis
```python
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits a given text into smaller chunks.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True # Hinweis
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        raise

# Kommentar
# long_text = "Kommentar"
# chunks = split_text_into_chunks(long_text, chunk_size=50, chunk_overlap=10)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}: {chunk}")
```

## Kommentar
`/backend/app/tools/split_tools.py`