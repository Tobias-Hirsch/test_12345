import httpx
from typing import List, Dict, Any
from ..core.config import settings  # Global import for configuration settings

# Assuming OLLAMA_URL and OLLAMA_COT_MODE are available from environment variables
OLLAMA_URL = settings.OLLAMA_SERVING_URL # Use the specific serving URL
OLLAMA_CHAT_MODEL = settings.OLLAMA_CHAT_MODEL # Use the chat model from COT_MODE
OLLAMA_COT_MODEL = settings.OLLAMA_COT_MODEL

def _sanitize_ollama_options(options: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Ollama does not understand several OpenAI-style option names.
    Keep only options accepted by Ollama and map max_tokens -> num_predict.
    """
    if not options:
        return {}

    sanitized: Dict[str, Any] = {}

    if "num_predict" in options and options["num_predict"] is not None:
        sanitized["num_predict"] = options["num_predict"]
    elif "max_tokens" in options and options["max_tokens"] is not None:
        sanitized["num_predict"] = options["max_tokens"]

    allowed_keys = {
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "seed",
        "stop",
        "num_ctx",
        "num_predict",
    }

    for key, value in options.items():
        if key in allowed_keys and value is not None:
            sanitized[key] = value

    return sanitized

async def get_chat_response(messages: List[Dict[str, str]]) -> str:
    """
    Sends a list of messages to the Ollama chat API and returns the bot's response.

    Args:
        messages: A list of message dictionaries, each with 'role' and 'content'.
                  Example: [{'role': 'user', 'content': 'Hello!'}]

    Returns:
        The content of the bot's response message, or an error message if the request fails.
    """
    api_endpoint = f"{OLLAMA_URL}/chat"
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False, # We want the full response at once
        "think": False,
    }
    sanitized_options = _sanitize_ollama_options({
        # "num_ctx": 4096,        #TODO Anpassen!!!!
        "num_predict": settings.OLLAMA_NUM_PREDICT,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
    })
    if sanitized_options:
        payload["options"] = sanitized_options

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_endpoint, json=payload, timeout=60.0) # Add a timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            response_data = response.json()
            # Extract the content from the bot's message
            if 'message' in response_data and 'content' in response_data['message']:
                return response_data['message']['content']
            else:
                print(f"Unexpected response format from Ollama API: {response_data}")
                return "Error: Received unexpected response from AI."

    except httpx.RequestError as e:
        print(f"HTTP request failed while calling Ollama API: {e}")
        return f"Error: Could not connect to AI service. Details: {e}"
    except httpx.HTTPStatusError as e:
        print(f"HTTP error response from Ollama API: {e.response.status_code} - {e.response.text}")
        return f"Error: AI service returned an error. Status: {e.response.status_code}"
    except Exception as e:
        print(f"An unexpected error occurred while calling Ollama API: {e}")
        return f"Error: An unexpected error occurred with the AI service. Details: {e}"

async def get_chat_response_stream(messages: List[Dict[str, str]], show_think_process: bool = False, options: Dict[str, Any] = None):
    """
    Sends a list of messages to the Ollama chat API and yields the bot's response chunks.

    Args:
        messages: A list of message dictionaries, each with 'role' and 'content'.
        show_think_process: A boolean to determine which model to use.
        options: An optional dictionary of Ollama options (e.g., temperature).

    Returns:
        An async generator that yields chunks of the bot's response content.
    """
    model= OLLAMA_COT_MODEL if show_think_process else OLLAMA_CHAT_MODEL

    # --- FINAL ATTEMPT: Hardcoding all parameters to ensure they are sent correctly ---
    api_endpoint = f"{OLLAMA_URL}/chat" # Revert to /chat endpoint
    #"num_predict": 4096,
    # Hardcode all parameters to eliminate any variables from settings or options
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
    }
    sanitized_options = _sanitize_ollama_options(options)
    if sanitized_options:
        payload["options"] = sanitized_options

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", api_endpoint, json=payload, timeout=300.0) as response:
                response.raise_for_status()
                async for chunk in response.aiter_lines():
                    if chunk:
                        try:
                            import json
                            data = json.loads(chunk)
                            if 'message' in data and 'content' in data['message']:
                                yield data['message']['content']
                            elif 'done' in data and data['done']:
                                pass
                        except json.JSONDecodeError:
                            pass
                        except Exception:
                            pass

    except httpx.RequestError as e:
        yield f"Error: Could not connect to AI service. Details: {e}"
    except httpx.HTTPStatusError as e:
        error_text = e.response.text
        print(f"HTTP error response from Ollama API: {e.response.status_code} - {error_text}")
        yield f"Error: AI service returned an error. Status: {e.response.status_code}. Details: {error_text}"
    except Exception as e:
        yield f"Error: An unexpected error occurred with the AI service. Details: {e}"


# Example usage (for testing purposes, not part of the service logic)
# async def main():
#     test_messages = [
#         {'role': 'system', 'content': 'You are a helpful assistant. You always answer in a comprehensive manner and provide detailed explanations, your response always in not less than 1000 words.'},
#         {'role': 'user', 'content': 'what do you know about injection moulding buisiness?'}
#     ]
#     print("AI Response: ", end="")
#     async for chunk in get_chat_response_stream(test_messages):
#         print(chunk, end="", flush=True)
#     print()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())