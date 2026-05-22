import re
from typing import List
import asyncio
from ..core.config import settings  # Global import for configuration settings

AI_TEMPLATE_SEGMENT_SPLIT_MAX_SIZE = int(settings.AI_TEMPLATE_SEGMENT_SPLIT_MAX_SIZE)

async def semantic_text_splitter(text: str, max_length: int = 2000) -> List[str]:
    """
    Hinweis

    Hinweis
        text (str): Hinweis
        max_length (int): Hinweis

    Hinweis
        list: Hinweis
    """
    # Kommentar
    if len(text) <= max_length:
        return [text]

    # Kommentar
    segments = []
    current_segment = ""

    # Kommentar
    paragraphs = re.split(r'\n\s*\n', text)

    # Kommentar
    tasks = []

    for paragraph in paragraphs:
        # Kommentar
        if len(paragraph) > max_length:
            # Kommentar
            tasks.append(_process_long_paragraph(paragraph, max_length))
        else:
            # Kommentar
            if len(current_segment) + len(paragraph) + 2 <= max_length:  # +2 for newline
                if current_segment:
                    current_segment += "\n\n" + paragraph
                else:
                    current_segment = paragraph
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = paragraph

    # Kommentar
    if tasks:
        long_paragraph_results = await asyncio.gather(*tasks)
        # Kommentar
        for result in long_paragraph_results:
            segments.extend(result)

    # Kommentar
    if current_segment:
        segments.append(current_segment.strip())

    return segments


async def _process_long_paragraph(paragraph: str, max_length: int) -> List[str]:
    """
    Hinweis

    Hinweis
        paragraph (str): Hinweis
        max_length (int): Hinweis

    Hinweis
        list: Hinweis
    """
    segments = []
    current_segment = ""

    # Kommentar
    sentence_endings = r'([. !?\.\!\?][\"\'\'\"]?)'
    paragraph_sentences = re.split(sentence_endings, paragraph)

    # Kommentar
    sentences = []
    i = 0
    while i < len(paragraph_sentences):
        if i + 1 < len(paragraph_sentences) and re.match(sentence_endings, paragraph_sentences[i + 1]):
            sentences.append(paragraph_sentences[i] + paragraph_sentences[i + 1])
            i += 2
        else:
            if paragraph_sentences[i].strip():
                sentences.append(paragraph_sentences[i])
            i += 1

    for sentence in sentences:
        # Kommentar
        if len(sentence) > max_length:
            await _process_long_sentence(sentence, max_length, segments)
        else:
            # Kommentar
            if len(current_segment) + len(sentence) <= max_length:
                current_segment += sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence

    # Kommentar
    if current_segment:
        segments.append(current_segment.strip())

    return segments


async def _process_long_sentence(sentence: str, max_length: int, segments: List[str]) -> None:
    """
    Hinweis

    Hinweis
        sentence (str): Hinweis
        max_length (int): Hinweis
        segments (List[str]): Hinweis
    """
    current_segment = ""
    punctuation_splits = re.split(r'([,;, ; , ])', sentence)
    i = 0
    while i < len(punctuation_splits):
        part = punctuation_splits[i]
        # Kommentar
        if i + 1 < len(punctuation_splits) and len(punctuation_splits[i + 1]) == 1:
            part += punctuation_splits[i + 1]
            i += 2
        else:
            i += 1

        if len(current_segment) + len(part) <= max_length:
            current_segment += part
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = part

    # Kommentar
    if current_segment:
        segments.append(current_segment.strip())