from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, List


SUMMARY_ARTIFACT_MARKERS = (
    "multiple items of type",
    "requires a single string output",
    "single string output",
    "mehreren zeichenfolgen",
    "mehrere zeichenfolgen",
    "ein einzelner string",
)


def normalize_summary_text(summary: Any) -> str:
    if not isinstance(summary, str):
        return "No reliable summary available."

    cleaned_summary = " ".join(summary.split()).strip()
    if not cleaned_summary:
        return "No reliable summary available."

    lowered = cleaned_summary.lower()
    if any(marker in lowered for marker in SUMMARY_ARTIFACT_MARKERS):
        return "No reliable summary available."

    return cleaned_summary


def sanitize_message_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content

    paragraphs = re.split(r"\n\s*\n", content)
    kept_paragraphs: List[str] = []

    for index, paragraph in enumerate(paragraphs):
        stripped_paragraph = paragraph.strip()
        lowered = stripped_paragraph.lower()

        if any(marker in lowered for marker in SUMMARY_ARTIFACT_MARKERS):
            if kept_paragraphs and re.fullmatch(r"(hinweis|note)\s*:?", kept_paragraphs[-1].strip(), re.IGNORECASE):
                kept_paragraphs.pop()
            continue

        if re.fullmatch(r"(hinweis|note)\s*:?", stripped_paragraph, re.IGNORECASE):
            next_paragraph = paragraphs[index + 1].strip().lower() if index + 1 < len(paragraphs) else ""
            if any(marker in next_paragraph for marker in SUMMARY_ARTIFACT_MARKERS):
                continue

        kept_paragraphs.append(stripped_paragraph)

    sanitized = "\n\n".join(paragraph for paragraph in kept_paragraphs if paragraph).strip()
    return sanitized or content


def sanitize_source_document(source: Any) -> Any:
    if not isinstance(source, dict):
        return source

    sanitized = deepcopy(source)
    if "summary" in sanitized:
        sanitized["summary"] = normalize_summary_text(sanitized.get("summary"))
    return sanitized


def sanitize_chat_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    sanitized = deepcopy(message)
    sanitized["content"] = sanitize_message_content(sanitized.get("content"))

    if isinstance(sanitized.get("source_documents"), list):
        sanitized["source_documents"] = [
            sanitize_source_document(source)
            for source in sanitized["source_documents"]
        ]

    search_results = sanitized.get("search_results")
    if isinstance(search_results, dict) and isinstance(search_results.get("source_documents"), list):
        search_results["source_documents"] = [
            sanitize_source_document(source)
            for source in search_results["source_documents"]
        ]

    # Keep older stored messages compatible with the current frontend shape.
    if sanitized.get("source_documents") and not (
        isinstance(search_results, dict) and search_results.get("source_documents")
    ):
        sanitized["search_results"] = sanitized.get("search_results") or {}
        sanitized["search_results"]["source_documents"] = deepcopy(sanitized["source_documents"])

    return sanitized


def sanitize_chat_history(history: Any) -> List[Dict[str, Any]]:
    if not isinstance(history, list):
        return []
    return [sanitize_chat_message(message) for message in history]


def sanitize_conversation_document(conversation: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = deepcopy(conversation)
    sanitized["messages"] = sanitize_chat_history(sanitized.get("messages", []))
    return sanitized
