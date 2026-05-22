import logging
import json
import io
from minio import Minio
from minio.error import S3Error
import asyncio
import os
from typing import Optional
from datetime import timedelta
from ..core.config import settings # Global import

# Kommentar
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

MINIO_ENDPOINT = settings.MINIO_ENDPOINT
MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
MINIO_BUCKET_NAME = settings.MINIO_BUCKET_NAME
logger.info(f"MinIO Endpoint loaded from env: {MINIO_ENDPOINT}")

def store_document_in_minio(file_path: str, object_name: str, bucket_name: str, minio_endpoint: Optional[str] = None) -> str:
    """
    Uploads a file to MinIO with a specified object name and bucket.

    Args:
        file_path: The local path to the file.
        object_name: The desired name for the object in MinIO.
        bucket_name: The name of the bucket to upload to.
        minio_endpoint: Optional. The MinIO endpoint to use. If None, uses MINIO_ENDPOINT from env.

    Returns:
        The URL of the uploaded object.
    """
    actual_minio_endpoint = minio_endpoint if minio_endpoint else MINIO_ENDPOINT
    if not actual_minio_endpoint:
        logger.error("MinIO endpoint is not defined.")
        raise ValueError("MinIO endpoint is not defined.")

    try:
        minio_client = Minio(
            actual_minio_endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Disable SSL for local development
        )

        # Check if bucket exists
        found = minio_client.bucket_exists(bucket_name)
        if not found:
            minio_client.make_bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' created.")
        else:
            logger.info(f"Bucket '{bucket_name}' already exists.")

        # Upload the file
        minio_client.fput_object(
            bucket_name, object_name, file_path,
        )

        # Construct the URL (adjust if your MinIO setup uses a different URL structure)
        minio_url = f"http://{actual_minio_endpoint}/{bucket_name}/{object_name}"
        logger.info(f"Successfully uploaded '{file_path}' to '{bucket_name}/{object_name}'. URL: {minio_url}")
        return minio_url

    except S3Error as e:
        logger.error(f"Error occurred during MinIO upload: {e}")
        raise

def generate_presigned_url(bucket_name: str, object_name: str, expires_in_seconds: int = 3600) -> Optional[str]:
    """
    Generates a presigned URL for an object in MinIO.

    Args:
        bucket_name: The name of the bucket.
        object_name: The name of the object.
        expires_in_seconds: The expiration time in seconds for the URL. Defaults to 1 hour.

    Returns:
        The presigned URL, or None if an error occurs.
    """
    try:
        minio_client = Minio(
            MINIO_ENDPOINT, # Use the environment endpoint for generating presigned URLs
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # Get a presigned URL for the object
        url = minio_client.presigned_get_object(
            bucket_name,
            object_name,
            expires=timedelta(seconds=expires_in_seconds)
        )
        logger.info(f"Generated presigned URL for '{object_name}': {url}")
        return url
    except S3Error as e:
        logger.error(f"Error generating presigned URL for '{object_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while generating presigned URL: {e}")
        return None

def get_document_from_minio(object_name: str, bucket_name: str, dest_path: str):
    """
    Downloads a document from MinIO.

    Args:
        object_name: The name of the object in the MinIO bucket.
        dest_path: The local path to save the downloaded file.

    Returns:
        The local path where the file was saved.
    """
    try:
        minio_client = Minio(
            MINIO_ENDPOINT, # Use the environment endpoint
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Disable SSL for local development
        )

        minio_client.fget_object(
            bucket_name,
            object_name,
            dest_path,
        )
        logger.info(f"Successfully downloaded '{object_name}' to '{dest_path}'")
        return dest_path

    except S3Error as e:
        logger.error(f"Error occurred while downloading '{object_name}': {e}")
        raise

def delete_document_from_minio(object_name: str, bucket_name: str):
    """
    Deletes a document from a specified bucket in MinIO.

    Args:
        object_name: The name of the object in the MinIO bucket.
        bucket_name: The name of the bucket to delete from.
    """
    try:
        minio_client = Minio(
            MINIO_ENDPOINT, # Use the environment endpoint
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Disable SSL for local development
        )

        minio_client.remove_object(
            bucket_name,
            object_name,
        )
        logger.info(f"Successfully deleted '{object_name}' from '{bucket_name}'")

    except S3Error as e:
        logger.error(f"Error occurred while deleting '{object_name}' from bucket '{bucket_name}': {e}")
        raise

async def get_document_bytes_from_minio(object_name: str, bucket_name: str) -> bytes:
    """
    Downloads a document from MinIO and returns its content as bytes.
    This is an async function that runs the synchronous MinIO call in a thread.

    Args:
        object_name: The name of the object in the MinIO bucket.
        bucket_name: The name of the bucket where the object is stored.

    Returns:
        The content of the file as bytes.
    """
    
    def _get_bytes():
        try:
            minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=False
            )
            
            response = minio_client.get_object(bucket_name, object_name)
            file_bytes = response.read()
            return file_bytes
        except S3Error as e:
            logger.error(f"Error occurred while getting bytes for '{object_name}': {e}")
            raise
        finally:
            if 'response' in locals() and response:
                response.close()
                response.release_conn()

    # Run the synchronous MinIO operation in a separate thread
    # to avoid blocking the asyncio event loop.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _get_bytes)

def store_json_object_in_minio(data: dict, object_name: str, bucket_name: str) -> Optional[str]:
    """
    Uploads a Python dictionary as a JSON file to MinIO.

    Args:
        data: The dictionary (JSON object) to upload.
        object_name: The desired name for the object in MinIO (e.g., 'my-data.json').
        bucket_name: The name of the bucket to upload to.

    Returns:
        The object name if successful, otherwise None.
    """
    if not MINIO_ENDPOINT:
        logger.error("MinIO endpoint is not defined.")
        return None

    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        # Ensure bucket exists
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' created.")

        # Convert dict to JSON bytes and upload
        json_bytes = json.dumps(data, indent=4, ensure_ascii=False).encode('utf-8')
        json_stream = io.BytesIO(json_bytes)
        
        minio_client.put_object(
            bucket_name,
            object_name,
            json_stream,
            length=len(json_bytes),
            content_type='application/json'
        )
        
        logger.info(f"Successfully uploaded JSON object '{object_name}' to '{bucket_name}'.")
        return object_name

    except S3Error as e:
        logger.error(f"Error occurred during MinIO JSON object upload: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during MinIO JSON object upload: {e}")
        return None

def read_json_object_from_minio(object_name: str, bucket_name: str) -> Optional[dict]:
    """
    Reads a JSON object from MinIO and returns it as a Python dictionary.

    Args:
        object_name: The name of the object in MinIO.
        bucket_name: The name of the bucket where the object is stored.

    Returns:
        A dictionary if successful, otherwise None.
    """
    if not MINIO_ENDPOINT:
        logger.error("MinIO endpoint is not defined.")
        return None

    response = None
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        response = minio_client.get_object(bucket_name, object_name)
        json_bytes = response.read()
        
        data = json.loads(json_bytes)
        logger.info(f"Successfully read JSON object '{object_name}' from '{bucket_name}'.")
        return data

    except S3Error as e:
        logger.error(f"Error occurred while reading MinIO JSON object '{object_name}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from object '{object_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading MinIO JSON object: {e}")
        return None
    finally:
        if response:
            response.close()
            response.release_conn()