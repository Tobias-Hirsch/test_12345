import logging
import nltk
import json
import asyncio
from typing import List, AsyncGenerator, Callable
from app.core.config import settings
from app.llm.llm import get_llm

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, LangDetectException
except ImportError:
    logger.error("langdetect not installed. Please run 'pip install langdetect' to enable automatic language detection.")
    # Provide a fallback function if langdetect is not available
    def detect(text): return "en" # Default to English
    class LangDetectException(Exception): pass

# --- Text Splitter Implementations ---

def _split_text_with_seperator(text: str, separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Helper function to split text by a separator."""
    splits = text.split(separator)
    chunks, current_chunk = [], ""
    for i, split in enumerate(splits):
        if len(current_chunk) + len(split) + len(separator) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
        
        current_chunk += split + separator
        
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def recursive_character_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text recursively using a list of separators.
    Tries to split by paragraphs, then sentences, then words.
    """
    separators = ["\n\n", "\n", " ", ""]
    
    final_chunks = []
    
    # Start with the full text
    texts_to_split = [text]
    
    for sep in separators:
        new_texts_to_split = []
        for text_to_split in texts_to_split:
            if len(text_to_split) <= chunk_size:
                final_chunks.append(text_to_split)
                continue
            
            # Split the text and add the new smaller parts to be processed
            splits = _split_text_with_seperator(text_to_split, sep, chunk_size, chunk_overlap)
            new_texts_to_split.extend(splits)
        texts_to_split = new_texts_to_split

    final_chunks.extend(texts_to_split)
    return [chunk for chunk in final_chunks if chunk.strip()]


def sentence_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text by sentences and then groups them into chunks."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        logger.info("Downloading NLTK 'punkt' model for sentence splitting...")
        nltk.download('punkt', quiet=True)
        
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start the new chunk with some overlap from the previous one
            overlap_part = chunks[-1][-chunk_overlap:]
            current_chunk = overlap_part
        current_chunk += " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def get_text_splitter() -> Callable[[str, int, int], List[str]]:
    """Factory function to get the configured text splitter."""
    if settings.TEXT_SPLITTER_STRATEGY == "sentence":
        logger.info("Using NLP-based sentence text splitter.")
        return sentence_split
    
    logger.info("Using recursive character text splitter.")
    return recursive_character_split

# --- Streaming Map-Reduce Implementation ---

async def _recursive_summarize(summaries: List[str], llm_instance, lang: str, max_tokens: int = 3000) -> str:
    """
    (Tool) Recursively summarizes a list of summaries until the total token count is below a threshold.
    """
    combined_text = "\n\n".join(summaries)
    if len(combined_text) / 4 < max_tokens: # Simple token estimation
        return combined_text

    logger.info(f"Combined summaries length ({len(combined_text)}) exceeds threshold. Recursively summarizing...")
    
    mid_point = len(summaries) // 2
    batch_1 = summaries[:mid_point]
    batch_2 = summaries[mid_point:]

    prompt_template = {
        "en": "Please summarize the following collection of summaries concisely in English:\n\n{text}",
        "de": "Bitte fasse die folgende Sammlung von Zusammenfassungen prägnant auf Deutsch zusammen:\n\n{text}",
    }.get("de" if lang.startswith("de") else lang, "Please summarize the following collection of summaries concisely:\n\n{text}")

    summary1_task = llm_instance.ainvoke(prompt_template.format(text="\n\n".join(batch_1)))
    summary2_task = llm_instance.ainvoke(prompt_template.format(text="\n\n".join(batch_2)))
    
    results = await asyncio.gather(summary1_task, summary2_task)
    new_summaries = [res.content for res in results]
    
    return await _recursive_summarize(new_summaries, llm_instance, lang, max_tokens)


def _is_structured_data(text: str) -> bool:
    """Detect if text is structured data (like Excel/CSV content)."""
    # Look for table indicators
    table_indicators = ['Arbeitsblatt:', 'Unnamed:', '|', '\t', '---']
    nan_count = text.count('NaN')
    total_lines = len(text.split('\n'))
    
    # If more than 30% of content is NaN or has table indicators
    if nan_count > total_lines * 0.3 or any(indicator in text for indicator in table_indicators):
        return True
    return False

def _extract_table_summary(text: str) -> str:
    """Extract key information from structured table data without LLM."""
    lines = text.split('\n')
    summary_lines = []
    
    # Extract worksheet names
    worksheet_names = []
    for line in lines:
        if 'Arbeitsblatt:' in line or 'Sheet:' in line:
            worksheet_names.append(line.strip())
    
    # Count data rows (excluding headers and separators)
    data_rows = 0
    for line in lines:
        if line.strip() and not any(x in line for x in ['Arbeitsblatt:', '---', 'Unnamed:', 'Sheet:']):
            # Count lines that look like data (have numbers or meaningful content)
            if any(c.isdigit() or c in '.,+-' for c in line) and 'NaN' not in line:
                data_rows += 1
    
    # Extract column headers
    headers = []
    for line in lines:
        if 'Unnamed:' in line or ('|' in line and not line.startswith('---')):
            # Extract potential column names
            parts = line.split()
            for part in parts:
                if part not in ['NaN', '|', '---'] and len(part) > 2:
                    headers.append(part)
            break  # Only get first header line
    
    # Build summary
    if worksheet_names:
        summary_lines.extend(worksheet_names)
    
    if data_rows > 0:
        summary_lines.append(f"Datenzeilen: {data_rows}")
    
    if headers:
        summary_lines.append(f"Wichtige Spalten: {', '.join(headers[:5])}")  # First 5 columns
    
    # Add a sample of actual data
    sample_data = []
    for line in lines[:20]:  # First 20 lines
        if line.strip() and not any(x in line for x in ['Arbeitsblatt:', '---', 'Unnamed:']):
            if len(line.strip()) > 10 and 'NaN' not in line[:50]:  # Non-empty meaningful lines
                sample_data.append(line.strip()[:100])  # First 100 chars
                if len(sample_data) >= 3:
                    break
    
    if sample_data:
        summary_lines.append("Beispieldaten:")
        summary_lines.extend(sample_data)
    
    return '\n'.join(summary_lines) if summary_lines else text[:500]

async def summarize_long_text(text: str, show_think_process: bool = False) -> AsyncGenerator[str, None]:
    """
    (Tool) Summarizes long text using optimized strategies based on content type.
    For structured data, uses fast extraction. For unstructured text, uses LLM summarization.
    """
    llm_instance = get_llm(show_think_process=show_think_process)

    if not text or not text.strip():
        yield "Error: Document content is empty."
        return

    # Check if this is structured data (Excel/table content)
    if _is_structured_data(text):
        logger.info("Detected structured data, using fast extraction method")
        yield "Processing structured data...\n\n"
        
        # Use fast table summary extraction
        table_summary = _extract_table_summary(text)
        
        # Only use LLM for final formatting, not for processing each chunk
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "en"
        
        format_prompt = {
            "en": f"Please format and summarize the following table information clearly in English:\n\n{table_summary}",
            "de": f"Bitte formatiere und fasse die folgenden Tabelleninformationen klar auf Deutsch zusammen:\n\n{table_summary}"
        }.get("de" if lang.startswith("de") else lang, f"Please format and summarize the following table information clearly:\n\n{table_summary}")
        
        yield "Formatting summary...\n\n"
        final_summary = ""
        async for stream_chunk in llm_instance.astream(format_prompt):
            final_summary += stream_chunk.content
            yield stream_chunk.content
        
        yield f"__FINAL_SUMMARY_COMPLETE__:{final_summary}"
        return

    # Original logic for unstructured text
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
    except LangDetectException:
        lang = "en" # Default to English on detection failure
        logger.warning("Language detection failed. Defaulting to English.")

    prompts = {
        "en": {
            "chunk_summary": "Please summarize the following content concisely in English:\n\n{chunk}",
            "final_report": "Synthesize the following English summaries into a single, coherent final report in English. Structure it using Markdown.\n\n--- Collection of Summaries ---\n{final_context}"
        },
        "de": {
            "chunk_summary": "Bitte fasse den folgenden Inhalt prägnant auf Deutsch zusammen:\n\n{chunk}",
            "final_report": "Führe die folgenden deutschen Zusammenfassungen zu einem kohärenten Abschlussbericht auf Deutsch zusammen. Strukturiere die Antwort mit Markdown.\n\n--- Sammlung der Zusammenfassungen ---\n{final_context}"
        }
    }
    lang_prompts = prompts.get("de" if lang.startswith("de") else lang, prompts["en"])

    splitter = get_text_splitter()
    chunks = splitter(text, settings.TEXT_CHUNK_SIZE, settings.TEXT_CHUNK_OVERLAP)
    yield f"Analyzing document in '{lang}' ({len(chunks)} parts)...\n\n"

    summaries = []
    for i, chunk in enumerate(chunks):
        yield f"**Summary of Part {i+1}/{len(chunks)}:**\n"
        try:
            prompt = lang_prompts["chunk_summary"].format(chunk=chunk)
            summary_content = ""
            async for stream_chunk in llm_instance.astream(prompt):
                summary_content += stream_chunk.content
                yield stream_chunk.content
            summaries.append(summary_content)
            yield "\n\n"
        except Exception as e:
            logger.error(f"Error summarizing chunk {i+1}", exc_info=True)
            yield f"Error processing part {i+1}: {e}\n\n"

    if not summaries:
        yield "Error: Could not generate any summaries from the document parts."
        return

    yield "\n**All parts summarized. Generating final, converged summary...**\n\n"
    
    try:
        final_context = await _recursive_summarize(summaries, llm_instance, lang)
        final_prompt = lang_prompts["final_report"].format(final_context=final_context)
        
        # This is the final summary, we stream it chunk by chunk
        final_summary = ""
        async for stream_chunk in llm_instance.astream(final_prompt):
            final_summary += stream_chunk.content
            yield stream_chunk.content
        
        # The very last thing this generator does is yield the full final summary.
        # The caller will know the stream is over and this is the final product.
        yield f"__FINAL_SUMMARY_COMPLETE__:{final_summary}"

    except Exception as e:
        logger.error("Error during final summarization", exc_info=True)
        yield f"\nAn error occurred during the final summarization: {e}"