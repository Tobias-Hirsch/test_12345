import os
import requests
from minio import Minio
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
import urllib3

# Warnhinweis
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Kommentar
# Kommentar
load_dotenv()

# MinIO Kommentar
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT_URL", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "documents")

# MongoDB Kommentar
MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
MONGO_DB_NAME = os.getenv("MONGO_INITDB_DATABASE")
MONGO_COLLECTION_NAME = "files"

# Kommentar
BACKEND_API_URL = "http://localhost:8000/api/v1/embeddings/embed_files"

# Kommentar
LOCAL_FILE_PATH = "test_native.pdf"
OBJECT_NAME = f"test_files/{LOCAL_FILE_PATH}"

# --- Kommentar
print("--- Hinweis")
try:
    # Kommentar
    minio_client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # Hinweis
    )
    print("MinIO Hinweis")

    # Kommentar
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    print("MongoDB Hinweis")

except Exception as e:
    print(f"Hinweis{e}")
    exit(1)


def main():
    """Hinweis"""
    print("\n--- Hinweis")

    # 1. Kommentar
    if not os.path.exists(LOCAL_FILE_PATH):
        print(f"Fehler: Hinweis'{LOCAL_FILE_PATH}' HinweisägeHinweis")
        print("Hinweis")
        return

    # 2. Datei hochladenKommentar
    print(f"\n--- Hinweis'{LOCAL_FILE_PATH}' Hinweis'{MINIO_BUCKET_NAME}' ---")
    try:
        # Kommentar
        found = minio_client.bucket_exists(MINIO_BUCKET_NAME)
        if not found:
            minio_client.make_bucket(MINIO_BUCKET_NAME)
            print(f"Bucket '{MINIO_BUCKET_NAME}' Hinweis")

        minio_client.fput_object(
            MINIO_BUCKET_NAME, OBJECT_NAME, LOCAL_FILE_PATH,
        )
        print(f"Hinweis'{OBJECT_NAME}'")
    except Exception as e:
        print(f"Hinweis{e}")
        return

    # 3. Kommentar
    print(f"\n--- Hinweis")
    try:
        file_doc = {
            "filename": LOCAL_FILE_PATH,
            "object_name": OBJECT_NAME,
            "bucket_name": MINIO_BUCKET_NAME,
            "status": "uploaded",
            "file_type": "pdf", # Hinweis
        }
        result = collection.insert_one(file_doc)
        file_id = str(result.inserted_id)
        print(f"MongoDB Hinweis{file_id}")
    except Exception as e:
        print(f"MongoDB Hinweis{e}")
        # Kommentar
        try:
            minio_client.remove_object(MINIO_BUCKET_NAME, OBJECT_NAME)
        except Exception as cleanup_e:
            print(f"Hinweis{cleanup_e}")
        return

    # 4. Kommentar
    print(f"\n--- Hinweis")
    try:
        payload = {
            "file_ids": [file_id]
        }
        headers = {
            "Content-Type": "application/json"
        }
        print(f"Senden POST Hinweis{BACKEND_API_URL}")
        print(f"Payload: {payload}")

        response = requests.post(BACKEND_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Hinweis

        print("API Hinweis")
        print(f"Status: {response.status_code}")
        print(f"Hinweis{response.json()}")
        print("\n--- Hinweis")
        print("Hinweis")
        print("Hinweis")

    except requests.exceptions.RequestException as e:
        print(f"API Hinweis{e}")
        if e.response:
            print(f"Hinweis{e.response.status_code}")
            print(f"Hinweis{e.response.text}")
    except Exception as e:
        print(f"Hinweis{e}")


if __name__ == "__main__":
    main()