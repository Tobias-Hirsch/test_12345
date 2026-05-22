import os
from typing import List
from ..core.config import settings # Global import
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.llm.llm import get_llm
from pydantic import BaseModel, Field


QW_API_KEY = settings.QW_API_KEY

class SummarizedDoc(BaseModel):
    summarize: str = Field(description="Zusammenfassung")
class SummarizedDocKeyWord(BaseModel):
    key_word: List[str] = Field(description="Schlüsselwörter")

# Prompt templates remain global
prompt_summarize_doc = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Backend
    Du bist Experte im Industriebereich. Damit Benutzer geeignete Literatur auswählen können, sollst du eine möglichst präzise Zusammenfassung der Literatur erstellen.

# Task
    Deine Aufgabe ist es, den Artikel aus fachlicher Perspektive anhand von Titel und Haupttext zusammenzufassen. Die Zusammenfassung soll möglichst kurz sein, aber alle für die Nutzung relevanten Informationen enthalten.
            """,
        ),
        ("placeholder", "{title}"),
        ("placeholder", "{content}"),
    ]
)


prompt_summarize_doc_key_word = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Backend
    Du bist Experte im Rechtsbereich. Damit Bürger geeignete Rechtstexte auswählen können, sollst du Schlüsselwörter für juristische Dokumente erzeugen.

# Task
    Deine Aufgabe ist es, aus fachjuristischer Perspektive anhand von Titel und Haupttext mehrere Schlüsselwörter zu bilden. Sie müssen die zentralen Informationen und betroffenen Bereiche möglichst vollständig abdecken, sensible Begriffe vermeiden und zwischen 5 und 150 Einträge umfassen.
            """,
        ),
        ("placeholder", "{title}"),
        ("placeholder", "{content}"),
    ]
)

# Chains are now constructed inside the functions to use the correct LLM instance.

async def fn_async_summarize_doc(title,content):
    llm_instance = get_llm(show_think_process=False)
    chain = prompt_summarize_doc | llm_instance.with_structured_output(SummarizedDoc).with_retry(stop_after_attempt=3)
    summarize_doc = await chain.ainvoke({"title": [("user", title)], "content": [("user", content)]},
            config={"run_name": f"chain_summarize_doc_{title}"})
    return summarize_doc

def fn_summarize_doc(title,content):
    llm_instance = get_llm(show_think_process=False)
    chain = prompt_summarize_doc | llm_instance.with_structured_output(SummarizedDoc).with_retry(stop_after_attempt=3)
    summarize_doc = chain.invoke({"title": [("user", title)], "content": [("user", content)]},
            config={"run_name": f"chain_summarize_doc_{title}"})
    return summarize_doc

async def fn_async_summarize_doc_key_word(title,content):
    llm_instance = get_llm(show_think_process=False)
    chain = prompt_summarize_doc_key_word | llm_instance.with_structured_output(SummarizedDocKeyWord).with_retry(stop_after_attempt=3)
    summarize_doc_key_word = await chain.ainvoke({"title": [("user", title)], "content": [("user", content)]},
            config={"run_name": f"chain_summarize_doc_key_word_{title}"})
    return summarize_doc_key_word

def fn_summarize_doc_key_word(title,content):
    llm_instance = get_llm(show_think_process=False)
    chain = prompt_summarize_doc_key_word | llm_instance.with_structured_output(SummarizedDocKeyWord).with_retry(stop_after_attempt=3)
    summarize_doc_key_word = chain.invoke({"title": [("user", title)], "content": [("user", content)]},
            config={"run_name": f"chain_summarize_doc_key_word_{title}"})
    return summarize_doc_key_word

