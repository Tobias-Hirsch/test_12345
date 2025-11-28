import os
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from minio import Minio
from minio.error import S3Error
from datetime import timedelta
from pymongo import MongoClient # Import MongoClient
from app.modules.mongodb_module import get_mongo_client # Import get_mongo_client from mongodb_module
from app.rag_knowledge.generic_knowledge import delete_milvus_data_by_filepath, delete_mongo_data_by_filename
from app.services.file_upload_service import get_minio_client
from ..core.config import settings # Global import
from app.models.database import FileGist, RagData
from app.models import database
from app.schemas import schemas
import logging
import os

logger = logging.getLogger(__name__)

async def purge_file_and_all_related_data(db: Session, file_id: int) -> bool:
    """
    Completely purges a file and all its related data across all systems:
    PostgreSQL, MinIO (source and intermediate), Milvus, and MongoDB.
    """
    db_file = db.query(FileGist).filter(FileGist.id == file_id).first()
    if not db_file:
        logger.warning(f"Purge operation failed: FileGist with ID {file_id} not found.")
        return False

    rag_entry = db_file.rag_data
    if not rag_entry:
        logger.error(f"Critical error: FileGist {file_id} has no associated RAG entry.")
        # Even without RAG info, we should attempt to clean up what we can.
        db.delete(db_file)
        db.commit()
        return False

    # --- Construct dynamic names ---
    dynamic_name = rag_entry.name.lower().replace(" ", "_")
    milvus_collection_name = f"rag_{dynamic_name}"
    mongo_db_name = f"rag_db_{dynamic_name}"
    mongo_collection_name = f"documents_{dynamic_name}"
    original_filename = db_file.filename
    original_filename_base = os.path.basename(original_filename)

    minio_client = None
    mongo_client = None

    try:
        # --- Initialize clients ---
        minio_client = get_minio_client()
        mongo_client = get_mongo_client()
        minio_bucket_name = settings.MINIO_BUCKET_NAME
        if not minio_bucket_name:
            raise ValueError("MINIO_BUCKET_NAME is not set.")

        # --- 1. Find and delete intermediate data from MinIO ---
        if mongo_client:
            mongo_db = mongo_client[mongo_db_name]
            mongo_collection = mongo_db[mongo_collection_name]
            mongo_doc = mongo_collection.find_one({"original_filename": original_filename_base})
            if mongo_doc and 'minio_object_path' in mongo_doc and mongo_doc['minio_object_path']:
                intermediate_path = mongo_doc['minio_object_path']
                try:
                    minio_client.remove_object(minio_bucket_name, intermediate_path)
                    logger.info(f"Successfully deleted intermediate MinIO object: {intermediate_path}")
                except S3Error as exc:
                    logger.warning(f"Could not delete intermediate MinIO object '{intermediate_path}': {exc}. Continuing cleanup.")

        # --- 2. Delete from Milvus ---
        await delete_milvus_data_by_filepath(milvus_collection_name, original_filename)
        logger.info(f"Successfully deleted data for '{original_filename}' from Milvus collection '{milvus_collection_name}'.")

        # --- 3. Delete from MongoDB ---
        await delete_mongo_data_by_filename(mongo_db_name, mongo_collection_name, original_filename_base)
        logger.info(f"Successfully deleted documents for '{original_filename_base}' from MongoDB '{mongo_db_name}/{mongo_collection_name}'.")

        # --- 4. Conditionally delete source file from MinIO ---
        if not db_file.is_third_party:
            try:
                minio_client.remove_object(minio_bucket_name, original_filename)
                logger.info(f"Successfully deleted first-party source file '{original_filename}' from MinIO.")
            except S3Error as e:
                logger.error(f"Failed to delete source file '{original_filename}' from MinIO: {e}. The database record will still be removed.")
        else:
            logger.info(f"Skipping MinIO source file deletion for third-party file '{original_filename}'.")

        # --- 5. Delete from PostgreSQL ---
        db.delete(db_file)
        db.commit()
        logger.info(f"Successfully purged FileGist record ID {file_id} and all associated data.")
        return True

    except Exception as e:
        logger.error(f"An error occurred during the purge operation for file ID {file_id}: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        if mongo_client:
            mongo_client.close()



def get_file_gist_with_download_url(db: Session, file_id: int) -> Optional[schemas.FileGist]:
    """
    Retrieves a FileGist by ID and generates a pre-signed MinIO download URL.
    """
    file_gist = db.query(FileGist).filter(FileGist.id == file_id).first()
    if not file_gist:
        return None

    minio_client = None
    minio_bucket_name = None

    try:
        if file_gist.is_third_party:
            logger.info(f"File ID {file_id} is a third-party file. Using CUSTOMER_MINIO settings.")
            # For third-party files, use the customer-specific MinIO settings
            endpoint = settings.CUSTOMER_MINIO_ENDPOINT
            access_key = settings.CUSTOMER_MINIO_ACCESS_KEY
            secret_key = settings.CUSTOMER_MINIO_SECRET_KEY
            minio_bucket_name = settings.CUSTOMER_MINIO_BUCKET_NAME
            
            if not all([endpoint, access_key, secret_key, minio_bucket_name]):
                raise ValueError("CUSTOMER MinIO settings are not fully configured.")

            # Create a new Minio client with third-party credentials
            # Note: endpoint might contain http/https, so we need to parse it.
            secure = "https" in endpoint
            clean_endpoint = endpoint.replace("https://", "").replace("http://", "")
            minio_client = Minio(clean_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        else:
            logger.info(f"File ID {file_id} is a first-party file. Using default MINIO settings.")
            # For internal files, use the default MinIO client
            minio_client = get_minio_client()
            minio_bucket_name = settings.MINIO_BUCKET_NAME
            if not minio_bucket_name:
                raise ValueError("Default MINIO_BUCKET_NAME is not set.")

        # Generate the pre-signed URL using the appropriate client and bucket
        download_url = minio_client.presigned_get_object(
            minio_bucket_name,
            file_gist.filename,
            expires=timedelta(days=7),  # URL valid for 7 days
        )
        file_gist_schema = schemas.FileGist.from_orm(file_gist)
        file_gist_schema.download_url = download_url
        return file_gist_schema

    except (S3Error, ValueError) as e:
        logger.error(f"MinIO Error generating pre-signed URL for file ID {file_id}: {e}", exc_info=True)
        return schemas.FileGist.from_orm(file_gist)  # Return without URL on error
    except Exception as e:
        logger.error(f"An unexpected error occurred during MinIO URL generation for file ID {file_id}: {e}", exc_info=True)
        return schemas.FileGist.from_orm(file_gist)  # Return without URL on error


def get_file_gists_with_download_urls(db: Session, file_ids: List[int]) -> List[schemas.FileGist]:
    """
    Retrieves a list of FileGists by IDs and generates pre-signed MinIO download URLs for each.
    """
    file_gists = db.query(FileGist).filter(FileGist.id.in_(file_ids)).all()
    
    minio_bucket_name = settings.MINIO_BUCKET_NAME
    if not minio_bucket_name:
        print("MINIO_BUCKET_NAME environment variable not set. Cannot generate download URLs.")
        return [schemas.FileGist.from_orm(fg) for fg in file_gists] # Return without URLs

    try:
        minio_client = get_minio_client()
        result_gists = []
        for file_gist in file_gists:
            try:
                download_url = minio_client.presigned_get_object(
                    minio_bucket_name,
                    file_gist.filename,
                    expires=timedelta(days=7),
                )
                file_gist_schema = schemas.FileGist.from_orm(file_gist)
                file_gist_schema.download_url = download_url
                result_gists.append(file_gist_schema)
            except S3Error as e:
                print(f"MinIO Error generating pre-signed URL for file ID {file_gist.id}: {e}")
                result_gists.append(schemas.FileGist.from_orm(file_gist)) # Add without URL
            except Exception as e:
                print(f"An unexpected error occurred during MinIO URL generation for file ID {file_gist.id}: {e}")
                result_gists.append(schemas.FileGist.from_orm(file_gist)) # Add without URL
        return result_gists
    except ValueError as e:
        print(f"MinIO client initialization error: {e}")
        return [schemas.FileGist.from_orm(fg) for fg in file_gists] # Return without URLs
    except Exception as e:
        print(f"An unexpected error occurred during batch MinIO URL generation: {e}")
        return [schemas.FileGist.from_orm(fg) for fg in file_gists] # Return without URLs

def get_rag_item_by_id(db: Session, rag_id: int) -> Optional[RagData]:
    """Fetches a RAG knowledge base item by its ID."""
    return db.query(RagData).filter(RagData.id == rag_id).first()

def get_rag_item_by_name(db: Session, name: str) -> Optional[RagData]:
    """Fetches a RAG knowledge base item by its name."""
    return db.query(RagData).filter(RagData.name == name).first()

def create_rag_item(db: Session, name: str, description: str) -> Optional[RagData]:
    """
    Creates a new RAG knowledge base item, assigning ownership to a default admin user.
    """
    # Check if a RAG item with this name already exists to prevent duplicates
    if get_rag_item_by_name(db, name):
        logger.warning(f"Attempted to create a RAG item with a name that already exists: {name}")
        return None

    # Find the first user to assign as the owner. Typically, user with ID 1 is the admin.
    # A more robust solution could be to look for a user with a specific 'admin' role.
    owner = db.query(database.User).filter(database.User.id == 1).first()
    if not owner:
        logger.error("Cannot create RAG item: Default owner (user with ID 1) not found.")
        return None

    try:
        db_rag_data = RagData(
            name=name,
            description=description,
            is_active=1,
            owner_id=owner.id
        )
        db.add(db_rag_data)
        db.commit()
        db.refresh(db_rag_data)
        logger.info(f"Successfully created new RAG item '{name}' with owner '{owner.username}'.")
        return db_rag_data
    except Exception as e:
        logger.error(f"Failed to create RAG item '{name}' in database: {e}", exc_info=True)
        db.rollback()
        return None

def get_all_rag_items(db: Session) -> List[RagData]:
    """Fetches all RAG knowledge base items."""
    return db.query(RagData).all()

def get_all_file_gists_for_rag_item(db: Session, rag_id: int, is_third_party: bool = False) -> List[FileGist]:
    """
    Retrieves all file gists for a specific RAG item, optionally filtering by
    the is_third_party flag.
    """
    query = db.query(FileGist).filter(FileGist.rag_id == rag_id)
    if is_third_party:
        query = query.filter(FileGist.is_third_party == True)
    return query.all()

def create_or_update_file_gist(db: Session, rag_id: int, filename: str, file_path_in_minio: str, etag: str, is_third_party: bool):
    """
    Creates a new file gist or updates an existing one based on filename and RAG ID.
    """
    # gist is deprecated, file_path_in_minio is the new field
    existing_gist = db.query(FileGist).filter(
        FileGist.rag_id == rag_id,
        FileGist.filename == filename
    ).first()

    if existing_gist:
        logger.info(f"Updating existing file gist for {filename} in RAG ID {rag_id}.")
        existing_gist.etag = etag
        existing_gist.gist = file_path_in_minio # Update gist to be the path
        existing_gist.processing_status = 'pending' # Reset status for re-processing
        existing_gist.processing_details = 'File has been modified and is queued for re-processing.'
    else:
        logger.info(f"Creating new file gist for {filename} in RAG ID {rag_id}.")
        new_gist = FileGist(
            rag_id=rag_id,
            filename=filename,
            gist=file_path_in_minio, # Use path for gist
            etag=etag,
            is_third_party=is_third_party,
            processing_status='pending',
            processing_details='New file detected and queued for processing.'
        )
        db.add(new_gist)
    
    db.commit()
