import ollama
import requests
from typing import List
from io import BytesIO
from PIL import Image
import json
from ..core.config import settings # Global import


OLLAMA_HOST = settings.OLLAMA_HOST
OLLAMA_PORT = settings.OLLAMA_PORT
OLLAMA_MODEL = settings.OLLAMA_MODEL  # Replace with the actual model name
OLLAMA_EMBEDDING_MODEL= settings.OLLAMA_EMBEDDING_MODEL
OLLAMA_RERANK_MODEL= settings.OLLAMA_RERANK_MODEL
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api"
ollama.api_url = OLLAMA_BASE_URL

def summarize_image_content(image_path: str) -> str:
    try:
        # Download the image
        response = requests.get(image_path, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Open the image using PIL
        image = Image.open(BytesIO(response.content))

        # Call the Ollama API
        ollama_response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt="Describe the image in detail.",
            images=[image]
        )

        # Extract the summary from the response
        summary = ollama_response['response']
        return summary

    except Exception as e:
        print("Error occurred: ", e)
        return f"Error summarizing image: {str(e)}"

def generate_embedding(text: str) -> List[float]:
    # Generate embedding for the given text using Ollama
    try:
        response = ollama.embeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print("Error occurred: ", e)
        return []
    

def is_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags" , timeout=2)

        return response.status_code == 200
    except requests.RequestException:
        return False
    
# Kommentar
def get_llm() -> List[str]:
    try:
        if not is_ollama_running():
            print("Warning: Ollama server is not running")
            return []
            
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return []
            
        if not response.content:
            print("Error: Empty response from API")
            return []
            
        result = response.json()
        llms = [
            llm["name"] for llm in result.get("models", [])
            if "code" not in llm["name"] and "embed" not in llm["name"] and "rerank" not in llm["name"]
        ]
        return llms
        
    except requests.RequestException as e:
        print(f"Connection error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def get_embeding_model() -> List[str]:
    try:
        if not is_ollama_running():
            print("Warning: Ollama server is not running")
            return []
            
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return []
            
        if not response.content:
            print("Error: Empty response from API")
            return []
            
        result = response.json()
        llms = [
            llm["name"] for llm in result.get("models", [])
            if "embed" in llm["name"]
        ]
        return llms
        
    except requests.RequestException as e:
        print(f"Connection error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
    

def get_rerank_model() -> List[str]:
    try:
        if not is_ollama_running():
            print("Warning: Ollama server is not running")
            return []
            
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=5)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return []
            
        if not response.content:
            print("Error: Empty response from API")
            return []
            
        result = response.json()
        llms = [
            llm["name"] for llm in result.get("models", [])
            if "rerank" in llm["name"]
        ]
        return llms
        
    except requests.RequestException as e:
        print(f"Connection error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def pull_model(model_name: str) -> bool:
    try:
        result = ollama.pull(model=model_name)
        return True
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")
        return False

def delete_model(model_name: str) -> bool:
    try:
        result = ollama.delete(model=model_name)
        return True
    except Exception as e:
        print(f"Error deleting model {model_name}: {e}")
        return False

def copy_model(source_model: str, dest_model: str) -> bool:
    try:
        result = ollama.copy(source=source_model, destination=dest_model)
        return True
    except Exception as e:
        print(f"Error copying model from {source_model} to {dest_model}: {e}")
        return False

def create_model(model_name: str, model_file: str) -> bool:
    try:
        result = ollama.create(model=model_name, modelfile=model_file)
        return True
    except Exception as e:
        print(f"Error creating model {model_name} from {model_file}: {e}")
        return False