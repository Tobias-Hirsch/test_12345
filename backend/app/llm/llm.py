import os
import base64
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from ..core.config import settings # Global import

QW_API_KEY = settings.QW_API_KEY
OLLAMA_QWEN_VL_MAX_LATEST = settings.OLLAMA_QWEN_VL_MAX_LATEST

# Ollama Configuration
OLLAMA_URL = settings.OLLAMA_SERVING_URL # Use the specific serving URL
OLLAMA_CHAT_MODEL = settings.OLLAMA_CHAT_MODEL # Use the chat model from COT_MODE
OLLAMA_COT_MODEL = settings.OLLAMA_COT_MODEL # Corrected variable name
OLLAMA_QWEN_MODEL = settings.OLLAMA_QWEN_MODEL # Added Qwen model for Ollama

def get_llm(show_think_process: bool = False) -> ChatOllama:
    """
    LLM factory function.
    Returns a ChatOllama instance based on the 'show_think_process' flag.
    - If True, uses the Chain-of-Thought model (OLLAMA_COT_MODEL).
    - If False, uses the standard chat model (OLLAMA_CHAT_MODEL).
    """
    if not OLLAMA_URL:
        # logger.error("OLLAMA_SERVING_URL is not set in the environment variables.")
        raise ValueError("OLLAMA_SERVING_URL must be set to a valid Ollama server URL.")

    if show_think_process:
        model_name = OLLAMA_COT_MODEL
        # logger.info(f"Using COT model: {model_name} at {OLLAMA_URL}")
    else:
        model_name = OLLAMA_CHAT_MODEL
        # logger.info(f"Using Chat model: {model_name} at {OLLAMA_URL}")
    
    # Robustly clean up the base URL to prevent issues with duplicate /api paths
    cleaned_base_url = OLLAMA_URL.rstrip("/")
    if cleaned_base_url.endswith("/api"):
        cleaned_base_url = cleaned_base_url[:-4]
    cleaned_base_url = cleaned_base_url.rstrip("/")

    return ChatOllama(model=model_name, base_url=cleaned_base_url)

# The global llm instance is now deprecated. Code should use the get_llm() factory.
# llm = ChatOpenAI(...)

async def llm_qwen_vl_max_ainvoke(message:str,url,model:str=OLLAMA_QWEN_VL_MAX_LATEST):
    client = OpenAI(
        api_key=QW_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model=model,
        # Kommentar
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "# backend\nFasse anhand der Benutzerfrage die relevanten Bildinhalte zusammen und übergib sie zur weiteren Beantwortung an nachgelagerte Agenten.\n# task\nDeine Aufgabe ist es, die für die Frage hilfreichen Schlüsselinformationen aus dem Bild zusammenzufassen. Gib das Ergebnis im JSON-Format aus und verwende keinen ```json```-Codeblock."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":url
                        },
                    },

                    {"type": "text", "text": message},
                ],
            },
        ],
    )
    print(completion)
    summarize_content = completion.choices[0].message.content.replace('"',"'").replace("\n","").replace(" ","").replace("{","").replace("}","")
    return summarize_content

async def llm_ollama_deepseek_ainvoke(message: str, model: str = OLLAMA_COT_MODEL, base_url: str = settings.OLLAMA_SERVING_URL) -> str:
    """
    Hinweis

    Args:
        message: BenutzerHinweis
        model: OllamaHinweis
        base_url: OllamaHinweis

    Returns:
        Hinweis
    """
    try:
        # client = OpenAI(
        #     base_url=OLLAMA_URL,  # Ollama expects the base URL without /api/
        #     # required but ignored
        #     api_key='ollama',
        #     model=model
        # )
        # Robustly clean up the base URL to prevent issues with duplicate /api paths
        cleaned_base_url = base_url.rstrip("/")
        if cleaned_base_url.endswith("/api"):
            cleaned_base_url = cleaned_base_url[:-4]
        cleaned_base_url = cleaned_base_url.rstrip("/")
        ollama_llm = ChatOllama(model=model, base_url=cleaned_base_url)
        # response = await client.chat.completions.create(
        #     messages=message
        #     )
        response = await ollama_llm.ainvoke(message)
        return response.content
    except Exception as e:
        print(f"Error calling Ollama model {model}: {e}")
        return f"Error generating response from Ollama: {e}"

async def llm_ollama_qwen_ainvoke(message: str, model: str = OLLAMA_QWEN_MODEL, base_url: str = settings.OLLAMA_SERVING_URL) -> str:
    """
    Hinweis

    Args:
        message: BenutzerHinweis
        model: OllamaHinweis
        base_url: OllamaHinweis

    Returns:
        Hinweis
    """
    try:
        # Robustly clean up the base URL to prevent issues with duplicate /api paths
        cleaned_base_url = base_url.rstrip("/")
        if cleaned_base_url.endswith("/api"):
            cleaned_base_url = cleaned_base_url[:-4]
        cleaned_base_url = cleaned_base_url.rstrip("/")
        ollama_llm = ChatOllama(model=model, base_url=cleaned_base_url)
        # For function calling, you would typically pass tools to the LLM
        # For now, we'll just return content, and the agent will handle parsing.
        response = await ollama_llm.ainvoke(message)
        return response.content
    except Exception as e:
        print(f"Error calling Ollama Qwen model {model}: {e}")
        return f"Error generating response from Ollama Qwen: {e}"

async def llm_ollama_vision_ainvoke(question: str, image_bytes: bytes, model: str = settings.OLLAMA_QWEN_VL_MAX_LATEST) -> str:
    """
    Hinweis

    Args:
        question: BenutzerHinweis
        image_bytes: Hinweis
        model: OllamaHinweis

    Returns:
        Hinweis
    """
    try:
        # Kommentar
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # Kommentar
        cleaned_base_url = settings.OLLAMA_SERVING_URL.rstrip("/")
        if cleaned_base_url.endswith("/api"):
            cleaned_base_url = cleaned_base_url[:-4]
        cleaned_base_url = cleaned_base_url.rstrip("/")
        
        llm = ChatOllama(model=model, base_url=cleaned_base_url)

        # Kommentar
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": question,
                },
                {
                    "type": "image_url",
                    "image_url": image_url,
                },
            ]
        )

        # Kommentar
        response = await llm.ainvoke([message])
        return response.content

    except Exception as e:
        print(f"Error calling Ollama vision model {model}: {e}")
        return f"Error generating response from Ollama Vision: {e}"