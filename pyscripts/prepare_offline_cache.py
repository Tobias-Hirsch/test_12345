import os
import sys
from pathlib import Path
import shutil

# --- Configuration ---
# Kommentar
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Kommentar
CACHE_DIR = Path("/home/cjb/.cache/huggingface/hub")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE_DIR)
os.environ['MODELSCOPE_CACHE'] = str(CACHE_DIR)

# Kommentar
try:
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfFolder
except ImportError:
    print("Error: 'mineru' or 'huggingface_hub' library not found.")
    print("Please install it on your host machine first: pip install mineru -U huggingface_hub")
    sys.exit(1)

def verify_snapshot(model_repo_id: str) -> bool:
    """Verifies that the snapshot directory and its contents exist."""
    # huggingface_hub < 0.22.0 uses a different cache layout
    # This code is for modern versions.
    model_path = CACHE_DIR / f"models--{model_repo_id.replace('/', '--')}"
    snapshots_path = model_path / "snapshots"
    if not snapshots_path.exists() or not any(snapshots_path.iterdir()):
        print(f"Verification FAILED: 'snapshots' directory is missing or empty for {model_repo_id}.")
        return False
    print(f"Verification PASSED: Found 'snapshots' for {model_repo_id}.")
    return True


def download_model(model_repo_id: str):
    """A helper function to download a single model and provide feedback."""
    try:
        print(f"--- Downloading model: {model_repo_id} ---")
        # Kommentar
        snapshot_download(repo_id=model_repo_id,
                          cache_dir=CACHE_DIR,
                          # local_files_only=False Kommentar
                          )
        
        if verify_snapshot(model_repo_id):
            print(f"--- Successfully downloaded and verified {model_repo_id} ---\n")
            return True
        else:
            print(f"!!! Download completed but verification failed for {model_repo_id} !!!\n")
            return False

    except Exception as e:
        print(f"!!! Failed to download model: {model_repo_id} !!!")
        print(f"Error: {e}")
        print("Please check your network connection and ensure you can access https://hf-mirror.com.")
        print("You might also need to check for proxy settings.\n")
        return False

def main():
    """Main function to download all required models for MinerU pipeline mode."""
    print("Starting offline model cache preparation for MinerU.")
    print(f"Using unified cache directory: {CACHE_DIR}")
    print(f"Using HuggingFace endpoint: {os.environ['HF_ENDPOINT']}")
    print("-" * 50)

    os.makedirs(CACHE_DIR, exist_ok=True)

    models_to_download = [
        "opendatalab/PDF-Extract-Kit-1.0",
        "opendatalab/yolo_v8_mfd",
    ]
    
    # KommentaröschenKommentar
    print("Recommendation: Before running, you may want to delete incomplete cache directories.")
    for repo_id in models_to_download:
        model_path = CACHE_DIR / f"models--{repo_id.replace('/', '--')}"
        if model_path.exists():
            print(f"Found existing cache for {repo_id} at {model_path}.")
            # shutil.rmtree(model_path)
            # print(f"Removed it to ensure a clean download.")


    all_successful = True
    for repo_id in models_to_download:
        if not download_model(repo_id):
            all_successful = False

    print("-" * 50)
    if all_successful:
        print("✅ All required models have been successfully downloaded and VERIFIED.")
        print("You can now restart the 'backend' service.")
    else:
        print("❌ Some models failed to download or verify. Please review the errors above.")

if __name__ == "__main__":
    main()
