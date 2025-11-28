import os
import uuid # Import uuid
import tempfile # Import tempfile
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.models.database import SessionLocal # Import SessionLocal for background tasks
import shutil
# from app.tools.pdf import process_pdf_content # This is a lingering, unused import and causes an error.
from app.rag_knowledge.generic_knowledge import (
    process_markdown_file, search_in_milvus, connect_to_milvus,
    get_embedding, insert_to_milvus, create_collection, get_mongo_client,
    summary_documents_content
)
from app.modules.minio_module import read_json_object_from_minio
# from app.services.document_processing_service import _process_document_from_bytes # This service no longer exists
from langchain.text_splitter import RecursiveCharacterTextSplitter
import aiofiles
from app.models.database import get_db, FileGist, RagData, User
from app.schemas import schemas
from app.services import auth
from app.dependencies.permissions import check_permission # Import the new permission checker
from app.modules.minio_module import store_document_in_minio
from minio import Minio
from minio.error import S3Error
from pymilvus import Collection
from app.rag_knowledge.generic_knowledge import delete_mongo_data_by_filename, delete_milvus_data_by_filepath
from ..core.config import settings

router = APIRouter()

@router.get("/query")
async def query_milvus(query_text: str = Query(..., description="The query string to search in Milvus"), current_user: User = Depends(auth.get_current_active_user)): # Add auth dependency
    try:
        collection_name = "markdown_data"
        await connect_to_milvus()
        collection = Collection(collection_name)

        search_results = await search_in_milvus(collection, query_text)

        return JSONResponse(content={"query": query_text, "results": search_results})
    except Exception as e:
        print(f"Error during Milvus query: {e}")
        raise HTTPException(status_code=500, detail=f"Error during Milvus query: {e}")

async def process_files_in_background(rag_id: int, file_ids: list[int], user_id: int):
    """
    This function runs in the background to process and embed files.
    It creates its own database session and includes comprehensive error logging.
    """
    db = SessionLocal()
    try:
        # The user is already authenticated by the time this task is called.
        # We fetch the user object again to ensure it's attached to this new session.
        current_user = db.query(User).filter(User.id == user_id).first()
        if not current_user:
            print(f"Background task failed: User with ID {user_id} not found.")
            return

        rag_entry = db.query(RagData).filter(RagData.id == rag_id).first()
        if not rag_entry:
            print(f"Background task failed: RAG entry with ID {rag_id} not found.")
            return

        # Construct dynamic names
        dynamic_name = rag_entry.name.lower().replace(" ", "_")
        milvus_collection_name = f"rag_{dynamic_name}"
        mongo_db_name = f"rag_db_{dynamic_name}"
        mongo_collection_name = f"documents_{dynamic_name}"

        # Retrieve FileGist entries
        file_gists = db.query(FileGist).filter(
            FileGist.id.in_(file_ids),
            FileGist.rag_id == rag_id
        ).all()

        if not file_gists:
            print(f"Background task: No valid files found to process for RAG ID {rag_id}.")
            return

        # Set initial status to "processing" for all files
        for fg in file_gists:
            fg.processing_status = 'processing'
            fg.processing_details = 'Task has started.'
        db.commit()

        try:
            minio_client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=False
            )
        except Exception as e:
            for fg in file_gists:
                fg.processing_status = 'failed'
                fg.processing_details = f'Failed to initialize MinIO client: {e}'
            db.commit()
            print(f"Background task failed: Could not initialize MinIO client. Error: {e}")
            return

        print(f"Background task started for RAG ID {rag_id} with {len(file_gists)} files.")

        for file_gist in file_gists:
            object_name = file_gist.filename
            original_file_extension = os.path.splitext(object_name)[1].lower()
            
            file_bytes = None
            response = None
            try:
                # Read file from MinIO into memory
                response = minio_client.get_object(settings.MINIO_BUCKET_NAME, object_name)
                file_bytes = response.read()
                print(f"BG Task: Read '{object_name}' into memory ({len(file_bytes)} bytes).")
                
                count = 0
                if original_file_extension == '.pdf':
                    from app.rag_knowledge.generic_knowledge import process_and_embed_pdf
                    count = await process_and_embed_pdf(
                        file_bytes=file_bytes,
                        original_filename=object_name,
                        milvus_collection_name=milvus_collection_name,
                        mongo_db_name=mongo_db_name,
                        rag_id=rag_id,
                        mongo_collection_name=mongo_collection_name
                    )
                elif original_file_extension in ['.md', '.txt']:
                    # For markdown, we still need a temporary file as the current logic expects a path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    
                    image_dir = tempfile.mkdtemp()
                    try:
                        count = await process_markdown_file(
                            tmp_path, image_dir, milvus_collection_name,
                            mongo_db_name, mongo_collection_name
                        )
                    finally:
                        os.remove(tmp_path)
                        shutil.rmtree(image_dir)
                else:
                    raise ValueError(f"Unsupported file type: {original_file_extension}")

                # If processing was successful
                file_gist.processing_status = 'success'
                file_gist.processing_details = f'Successfully processed {count} chunks.'
                print(f"BG Task: Successfully processed {file_gist.id} ({object_name}).")

            except Exception as e:
                import traceback
                error_str = f"Error processing file {object_name}: {e}"
                print(f"BG Task: --- ERROR ---")
                print(error_str)
                traceback.print_exc()
                print(f"BG Task: --- END ERROR ---")
                file_gist.processing_status = 'failed'
                file_gist.processing_details = str(e)
            
            finally:
                if response:
                    response.close()
                    response.release_conn()
                db.commit() # Commit status change for each file
        
        print(f"Background task for RAG ID {rag_id} completed.")

    except Exception as e:
        # This top-level exception catches errors like DB connection issues
        import traceback
        print(f"--- CRITICAL BACKGROUND TASK ERROR ---")
        print(f"Task: process_files_in_background")
        print(f"Arguments: rag_id={rag_id}, file_ids={file_ids}, user_id={user_id}")
        print(f"Exception: {e}")
        traceback.print_exc()
        print(f"--- END CRITICAL BACKGROUND TASK ERROR ---")
    finally:
        db.close()

def run_async_in_background(task, *args, **kwargs):
    """A synchronous wrapper to run an async task in a new event loop."""
    import asyncio
    try:
        # Create a new event loop for the background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Run the async task until it completes
        loop.run_until_complete(task(*args, **kwargs))
    finally:
        loop.close()

@router.post("/{rag_id}/embed_files")
async def embed_rag_files(
    rag_id: int,
    file_embed_request: schemas.FileEmbedRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
):
    """
    Triggers the embedding process for selected files as a background task.
    """
    # --- Perform initial, fast checks before starting the background task ---
    check_permission(db, current_user, "manage", resource_type="rag_data", resource_id=rag_id)

    rag_entry = db.query(RagData).filter(RagData.id == rag_id).first()
    if not rag_entry:
        raise HTTPException(status_code=404, detail="RAG entry not found")

    # Check if file_ids exist for the given rag_id
    found_files_count = db.query(FileGist).filter(
        FileGist.id.in_(file_embed_request.file_ids),
        FileGist.rag_id == rag_id
    ).count()

    if found_files_count != len(file_embed_request.file_ids):
        raise HTTPException(status_code=404, detail="One or more file IDs were not found for the specified RAG entry.")

    # --- Update status to 'pending' before adding to background ---
    db.query(FileGist).filter(
        FileGist.id.in_(file_embed_request.file_ids),
        FileGist.rag_id == rag_id
    ).update({"processing_status": "pending", "processing_details": "Awaiting background processing."}, synchronize_session=False)
    db.commit()

    # --- Add the long-running process to background tasks ---
    # We now wrap the async task in a synchronous function that handles the event loop.
    background_tasks.add_task(
        run_async_in_background,
        process_files_in_background,
        rag_id=rag_id,
        file_ids=file_embed_request.file_ids,
        user_id=current_user.id
    )

    # --- Return an immediate response ---
    return JSONResponse(
        status_code=202,
        content={"message": f"File processing for {len(file_embed_request.file_ids)} files has been queued."}
    )


@router.get("/{rag_id}/files/status")
def get_files_status(
    rag_id: int,
    file_ids: list[int] = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
):
    """
    Retrieves the processing status of one or more files.
    """
    check_permission(db, current_user, "read", resource_type="rag_data", resource_id=rag_id)

    file_gists = db.query(FileGist).filter(
        FileGist.id.in_(file_ids),
        FileGist.rag_id == rag_id
    ).all()

    if not file_gists:
        raise HTTPException(status_code=404, detail="No files found for the given IDs and RAG entry.")

    status_map = {
        gist.id: {
            "filename": gist.filename,
            "status": gist.processing_status,
            "details": gist.processing_details
        }
        for gist in file_gists
    }

    return status_map


@router.post("/{rag_id}/re_embed_files")
async def re_embed_rag_files(rag_id: int, file_embed_request: schemas.FileEmbedRequest, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_active_user)):
    """
    Triggers the re-embedding process. This is a destructive operation that first deletes
    all existing data (MinIO, Milvus, MongoDB) associated with the file, then re-processes it from scratch.
    """
    check_permission(db, current_user, "manage", resource_type="rag_data", resource_id=rag_id)
    
    rag_entry = db.query(RagData).filter(RagData.id == rag_id).first()
    if not rag_entry:
        raise HTTPException(status_code=404, detail="RAG entry not found")

    dynamic_name = rag_entry.name.lower().replace(" ", "_")
    milvus_collection_name = f"rag_{dynamic_name}"
    mongo_db_name = f"rag_db_{dynamic_name}"
    mongo_collection_name = f"documents_{dynamic_name}"

    file_gists = db.query(FileGist).filter(
        FileGist.id.in_(file_embed_request.file_ids),
        FileGist.rag_id == rag_id
    ).all()

    if len(file_gists) != len(file_embed_request.file_ids):
        raise HTTPException(status_code=404, detail="One or more file IDs not found for the specified RAG entry.")

    try:
        minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize MinIO client: {e}")

    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB.")

    processed_count = 0
    errors = []

    for file_gist in file_gists:
        original_filename = file_gist.filename
        original_filename_base = os.path.basename(original_filename)
        
        response = None
        try:
            print(f"--- Starting Re-embedding for: {original_filename} (FileGist ID: {file_gist.id}) ---")
            
            # --- Comprehensive Deletion Step ---
            print(f"  [1/3] Deleting data from Milvus and MongoDB...")
            await delete_milvus_data_by_filepath(milvus_collection_name, original_filename)
            
            mongo_db = mongo_client[mongo_db_name]
            mongo_collection = mongo_db[mongo_collection_name]
            mongo_doc = mongo_collection.find_one({"original_filename": original_filename_base})
            
            if mongo_doc and 'minio_object_path' in mongo_doc and mongo_doc['minio_object_path']:
                mineru_object_path = mongo_doc['minio_object_path']
                try:
                    minio_client.remove_object(settings.MINIO_BUCKET_NAME, mineru_object_path)
                    print(f"  [2/3] Deleted MinerU output from MinIO: {mineru_object_path}")
                except S3Error as exc:
                    print(f"  [WARNING] Could not delete MinerU output '{mineru_object_path}' from MinIO: {exc}")

            await delete_mongo_data_by_filename(mongo_db_name, mongo_collection_name, original_filename_base)
            
            # --- Re-processing Step ---
            print(f"  [3/3] Reading file from MinIO into memory and processing...")
            response = minio_client.get_object(settings.MINIO_BUCKET_NAME, original_filename)
            file_bytes = response.read()

            original_file_extension = os.path.splitext(original_filename)[1].lower()
            if original_file_extension == '.pdf':
                from app.rag_knowledge.generic_knowledge import process_and_embed_pdf
                count = await process_and_embed_pdf(
                    file_bytes=file_bytes,
                    original_filename=original_filename,
                    milvus_collection_name=milvus_collection_name,
                    mongo_db_name=mongo_db_name,
                    rag_id=rag_id,
                    mongo_collection_name=mongo_collection_name
                )
                processed_count += count
            elif original_file_extension in ['.md', '.txt']:
                with tempfile.NamedTemporaryFile(delete=False, suffix=original_file_extension) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                image_dir = tempfile.mkdtemp()
                try:
                    count = await process_markdown_file(
                        tmp_path, image_dir, milvus_collection_name,
                        mongo_db_name, mongo_collection_name
                    )
                    processed_count += count
                finally:
                    os.remove(tmp_path)
                    shutil.rmtree(image_dir)
            else:
                raise ValueError(f"Unsupported file type for re-embedding: {original_file_extension}")

            print(f"--- Successfully re-embedded: {original_filename} ---")

        except Exception as e:
            error_str = f"Error re-embedding file {original_filename}: {e}"
            print(f"--- ERROR: {error_str} ---")
            errors.append({"filename": original_filename, "error": str(e)})
        finally:
            if response:
                response.close()
                response.release_conn()

    if mongo_client:
        mongo_client.close()

    if errors:
        raise HTTPException(status_code=500, detail={"message": "Re-embedding completed with errors.", "processed_count": processed_count, "errors": errors})
    
    return JSONResponse(content={"message": f"Successfully re-embedded {len(file_gists)} files. Total chunks processed: {processed_count}."})


@router.post("/{rag_id}/retry_embedding")
async def retry_embedding_from_minio(
    rag_id: int,
    file_embed_request: schemas.FileEmbedRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
):
    """
    Retries the embedding process for a file that has already been processed by MinerU
    but failed in a subsequent step (e.g., embedding or DB insertion).
    This uses the persisted MinerU output from MinIO.
    """
    check_permission(db, current_user, "manage", resource_type="rag_data", resource_id=rag_id)

    rag_entry = db.query(RagData).filter(RagData.id == rag_id).first()
    if not rag_entry:
        raise HTTPException(status_code=404, detail="RAG entry not found")

    dynamic_name = rag_entry.name.lower().replace(" ", "_")
    milvus_collection_name = f"rag_{dynamic_name}"
    mongo_db_name = f"rag_db_{dynamic_name}"
    mongo_collection_name = f"documents_{dynamic_name}"

    file_gists = db.query(FileGist).filter(
        FileGist.id.in_(file_embed_request.file_ids),
        FileGist.rag_id == rag_id
    ).all()

    if len(file_gists) != len(file_embed_request.file_ids):
        raise HTTPException(status_code=404, detail="One or more file IDs not found for the specified RAG entry.")

    # MinIO client is not needed here as read_json_object_from_minio creates its own.
    
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB.")

    processed_count = 0
    errors = []

    try:
        mongo_db = mongo_client[mongo_db_name]
        mongo_collection = mongo_db[mongo_collection_name]

        for file_gist in file_gists:
            original_filename = file_gist.filename
            original_filename_base = os.path.basename(original_filename)
            
            try:
                print(f"--- Starting Embedding Retry for: {original_filename} ---")
                mongo_doc = mongo_collection.find_one({"original_filename": original_filename_base})

                if not mongo_doc or 'minio_object_path' not in mongo_doc or not mongo_doc['minio_object_path']:
                    raise ValueError("No MinerU output found in MinIO path. Please use the 're-embed' endpoint instead.")
                
                mineru_object_path = mongo_doc['minio_object_path']
                print(f"  [1/2] Found MinerU data at: {mineru_object_path}. Reading from MinIO...")

                mineru_data = read_json_object_from_minio(mineru_object_path, settings.MINIO_BUCKET)
                if not mineru_data:
                    raise ValueError(f"Failed to read or parse JSON from MinIO object: {mineru_object_path}")

                # Before re-embedding, clear out any potentially partial/stale data from Milvus for this file
                print(f"  [2/2] Clearing old vector data from Milvus and re-embedding...")
                await delete_milvus_data_by_filepath(milvus_collection_name, original_filename)

                # Local import to prevent circular dependency issues
                from app.rag_knowledge.generic_knowledge import embed_parsed_mineru_data
                count = await embed_parsed_mineru_data(
                    mineru_result=mineru_data,
                    original_filename=original_filename,
                    milvus_collection_name=milvus_collection_name,
                    mongo_db_name=mongo_db_name,
                    mongo_collection_name=mongo_collection_name,
                    minio_object_name=mineru_object_path # Pass the existing path
                )
                processed_count += count
                print(f"--- Successfully retried embedding for: {original_filename} ---")

            except Exception as e:
                error_str = f"Error retrying embedding for file {original_filename}: {e}"
                print(f"--- ERROR: {error_str} ---")
                errors.append({"filename": original_filename, "error": str(e)})
    
    finally:
        if mongo_client:
            mongo_client.close()

    if errors:
        raise HTTPException(status_code=500, detail={"message": "Retry completed with errors.", "processed_count": processed_count, "errors": errors})

    return JSONResponse(content={"message": f"Successfully retried embedding for {len(file_gists)} files. Total chunks processed: {processed_count}."})


@router.post("/files/{file_id}/resync", status_code=202)
async def resync_single_file(
    file_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
):
    """
    Triggers a non-destructive re-sync and embedding for a single file.
    This is the standard way to update a file's embeddings after a content change or processing error.
    """
    # --- 1. Fetch file and its RAG entry to check permissions ---
    db_file = db.query(FileGist).options(Session.joinedload(FileGist.rag)).filter(FileGist.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not db_file.rag:
        raise HTTPException(status_code=404, detail="Associated RAG item not found for this file.")

    # Use a comprehensive permission like "manage" for a re-embedding action
    check_permission(db, current_user, "manage", resource_type="rag_data", resource_id=db_file.rag_data.id)

    # --- 2. Update status to 'pending' to signal processing ---
    db_file.processing_status = "pending"
    db_file.processing_details = "Re-sync queued by user."
    db.commit()

    # --- 3. Add the processing task to the background ---
    background_tasks.add_task(
        run_async_in_background,
        process_files_in_background,
        rag_id=db_file.rag_id,
        file_ids=[file_id],
        user_id=current_user.id
    )

    # --- 4. Return an immediate response ---
    return {"message": f"Re-sync for file '{db_file.filename}' has been successfully queued."}
