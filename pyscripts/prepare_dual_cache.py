import os
import sys
from pathlib import Path
import shutil

# --- Configuration ---
# Kommentar
HF_CACHE_DIR = Path("/home/cjb/.cache/huggingface/hub")
MS_CACHE_DIR = Path("/home/cjb/.cache/modelscope/hub")

# Kommentar
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = str(HF_CACHE_DIR)

# Kommentar
os.environ['MODELSCOPE_CACHE'] = str(MS_CACHE_DIR)


# --- Sanity Checks ---
try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' or 'modelscope' library not found.")
    print("Please install them on your host machine first: pip install modelscope huggingface_hub -U")
    sys.exit(1)

def download_hf_model(repo_id: str):
    """Downloads a model from HuggingFace and verifies it."""
    print(f"--- Downloading HuggingFace model: {repo_id} ---")
    try:
        hf_snapshot_download(repo_id=repo_id, cache_dir=HF_CACHE_DIR)
        model_path = HF_CACHE_DIR / f"models--{repo_id.replace('/', '--')}"
        snapshots_path = model_path / "snapshots"
        if not snapshots_path.exists() or not any(snapshots_path.iterdir()):
            print(f"!!! Verification FAILED: 'snapshots' dir missing for {repo_id} !!!\n")
            return False
        print(f"--- Successfully downloaded and verified {repo_id} ---\n")
        return True
    except Exception as e:
        print(f"!!! FAILED to download {repo_id}: {e} !!!\n")
        return False

def download_ms_model(model_id: str):
    """Downloads a model from ModelScope."""
    print(f"--- Downloading ModelScope model: {model_id} ---")
    try:
        ms_snapshot_download(model_id=model_id)
        model_path = MS_CACHE_DIR / 'models' / model_id
        if not model_path.exists() or not any(model_path.iterdir()):
             print(f"!!! Verification FAILED: Directory missing for {model_id} !!!\n")
             return False
        print(f"--- Successfully downloaded and verified {model_id} ---\n")
        return True
    except Exception as e:
        print(f"!!! FAILED to download {model_id}: {e} !!!\n")
        return False

def main():
    print("Starting DUAL cache preparation for MinerU.")
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(MS_CACHE_DIR, exist_ok=True)
    
    print("\nStep 1: Downloading HuggingFace models...")
    hf_success = download_hf_model("opendatalab/PDF-Extract-Kit-1.0")

    print("\nStep 2: Downloading ModelScope models...")
    # KommentarätigenKommentar%Kommentar
    ms_success = download_ms_model("damo/MFD-yolov8-l")

    print("-" * 50)
    if hf_success and ms_success:
        print("✅ All required models have been successfully downloaded to their respective caches.")
        print("You can now proceed to the next step: updating docker-compose.prod.yml.")
    else:
        print("❌ Some models failed to download. Please review the errors above.")

if __name__ == "__main__":
    main()
