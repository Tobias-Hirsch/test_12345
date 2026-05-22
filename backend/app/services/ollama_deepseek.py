import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-coder"  # Oder deepseek-chat, Hinweis

def get_deepseek_reply(prompt: str) -> str:
    """
    Hinweis
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Ollama Kommentar{'response': '...'}
    return data.get("response", "").strip()