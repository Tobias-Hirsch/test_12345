import os
import aiofiles
import json
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends, status
import tempfile # Import tempfile
from typing import List, Dict # Import List and Dict
from fastapi.responses import JSONResponse, StreamingResponse, Response # Add StreamingResponse and Response
from sqlalchemy.orm import Session, joinedload # Import Session and joinedload
from datetime import timedelta
from urllib.parse import urlparse
import io # Add import for io
import logging
logger = logging.getLogger(__name__)
from app.rag_knowledge.generic_knowledge import process_markdown_file, search_in_milvus, connect_to_milvus, get_mongo_client
from pymilvus import utility
from app.tools.deal_document import get_text_from_uploaded_file
from app.llm.chain import fn_async_summarize_doc
from app.llm.llm import get_llm
import shutil # Import shutil for directory cleanup
import app.models.database as database
from app.models.database import get_db, FileGist, RagData, User
from app.schemas import schemas
from app.services import auth
from app.modules.minio_module import store_document_in_minio
from minio import Minio
from minio.error import S3Error
from app.dependencies.permissions import require_abac_permission, check_permission, has_permission
from app.services.query_filter_service import QueryFilterService, get_query_filter_service
from app.services.rag_file_service import purge_file_and_all_related_data
from ..core.config import settings
router = APIRouter()
 
# ... (deprecated /uploadfile/ endpoint remains commented out) ...

# RAG Data Endpoints
@router.post("", response_model=schemas.RagData, dependencies=[Depends(require_abac_permission("rag_data", "create"))], include_in_schema=False)
@router.post("/", response_model=schemas.RagData, dependencies=[Depends(require_abac_permission("rag_data", "create"))])
def create_rag_data(rag_data: schemas.RagDataCreate, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_active_user)):
    """
    Creates a new RAG data entry, based on ABAC policies.
    """
    db_rag_data = db.query(database.RagData).filter(database.RagData.name == rag_data.name).first()
    if db_rag_data:
        raise HTTPException(status_code=400, detail="RAG data with this name already exists")

    db_rag_data = database.RagData(
        name=rag_data.name,
        description=rag_data.description,
        is_active=1,
        owner_id=current_user.id
    )
    db.add(db_rag_data)
    db.commit()
    db.refresh(db_rag_data)
    return db_rag_data

@router.get("", response_model=schemas.RagDataListResponse, include_in_schema=False)
@router.get("/", response_model=schemas.RagDataListResponse)
def read_rag_data(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    qfs: QueryFilterService = Depends(get_query_filter_service),
    current_user: User = Depends(auth.get_current_active_user)
) -> schemas.RagDataListResponse:
    """
    Retrieves a list of RAG data entries the user has access to, based on ABAC policies.
    Also indicates if the user can create new RAG items.
    """
    filters = qfs.get_query_filters(resource_type="rag_data", action="read_list")
    query = db.query(database.RagData)
    if filters:
        query = query.filter(*filters)
    
    rag_items = query.offset(skip).limit(limit).all()
    
    # Check creation permission using the new boolean-returning function
    can_create = has_permission(
        db=db,
        redis_client=qfs.redis_client,
        current_user=current_user,
        action="create",
        resource_type="rag_data"
    )

    return schemas.RagDataListResponse(rag_data=rag_items, can_create=can_create)

@router.get("/{rag_id}", response_model=schemas.RagData)
def read_rag_data_by_id(rag_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_active_user)):
    """Retrieves a specific RAG data entry by ID, based on ABAC policies."""
    rag_data = db.query(database.RagData).filter(database.RagData.id == rag_id).first()
    if not rag_data:
        raise HTTPException(status_code=404, detail="RAG data not found")
        
    check_permission(db, current_user, "read", resource_type="rag_data", resource_id=rag_data.id)
    
    return rag_data

@router.put("/{rag_id}", response_model=schemas.RagData)
def update_rag_data(
    rag_id: int,
    rag_data_update: schemas.RagDataUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
) -> schemas.RagData:
    """Updates an existing RAG data entry, based on ABAC policies."""
    db_rag_data = db.query(database.RagData).filter(database.RagData.id == rag_id).first()
    if db_rag_data is None:
        raise HTTPException(status_code=404, detail="RAG data not found")

    check_permission(db, current_user, "update", resource_type="rag_data", resource_id=db_rag_data.id)

    for field, value in rag_data_update.model_dump(exclude_unset=True).items():
        setattr(db_rag_data, field, value)

    db.commit()
    db.refresh(db_rag_data)
    return db_rag_data

@router.delete("/{rag_id}")
def delete_rag_data(
    rag_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
) -> Dict[str, str]:
    """
    Deletes a RAG data entry and all associated data including:
    - Files in MinIO
    - FileGist records in PostgreSQL
    - The entire Milvus collection
    - The entire MongoDB database for that RAG item
    """
    db_rag_data = db.query(database.RagData).filter(database.RagData.id == rag_id).first()
    if db_rag_data is None:
        raise HTTPException(status_code=404, detail="RAG data not found")

    check_permission(db, current_user, "delete", resource_type="rag_data", resource_id=db_rag_data.id)

    rag_item_name = db_rag_data.name
    sanitized_rag_item_name = rag_item_name.lower().replace(" ", "_")
    milvus_collection_name = f"rag_{sanitized_rag_item_name}"
    mongo_db_name = f"rag_db_{sanitized_rag_item_name}"

    # 1. Delete associated files from MinIO and FileGist records from PostgreSQL
    associated_files = db.query(database.FileGist).filter(database.FileGist.rag_id == rag_id).all()
    if associated_files:
        minio_bucket_name = settings.MINIO_BUCKET_NAME
        if not minio_bucket_name:
            print("MinIO bucket name not set. Skipping MinIO deletion.")
        else:
            try:
                minio_client = Minio(settings.MINIO_ENDPOINT, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY, secure=False)
                for file_gist in associated_files:
                    # Only delete the object from MinIO if it's not a third-party file
                    if not file_gist.is_third_party:
                        try:
                            minio_client.remove_object(minio_bucket_name, file_gist.filename)
                            print(f"Successfully deleted '{file_gist.filename}' from MinIO.")
                        except S3Error as e:
                            print(f"Error deleting object '{file_gist.filename}' from Minio: {e}")
                    else:
                        print(f"Skipping deletion of third-party file '{file_gist.filename}' from MinIO.")
            except Exception as e:
                print(f"An unexpected error occurred during MinIO client initialization: {e}")
        
        for file_gist in associated_files:
            db.delete(file_gist)
        print(f"Deleted {len(associated_files)} FileGist records from PostgreSQL.")

    # 2. Delete the Milvus collection
    try:
        connect_to_milvus()
        if utility.has_collection(milvus_collection_name):
            utility.drop_collection(milvus_collection_name)
            print(f"Successfully dropped Milvus collection: {milvus_collection_name}")
        else:
            print(f"Milvus collection '{milvus_collection_name}' not found. Skipping.")
    except Exception as e:
        print(f"Error dropping Milvus collection '{milvus_collection_name}': {e}")

    # 3. Delete the MongoDB database
    mongo_client = None
    try:
        mongo_client = get_mongo_client()
        if mongo_client:
            mongo_client.drop_database(mongo_db_name)
            print(f"Successfully dropped MongoDB database: {mongo_db_name}")
    except Exception as e:
        print(f"Error dropping MongoDB database '{mongo_db_name}': {e}")
    finally:
        if mongo_client:
            mongo_client.close()

    # 4. Delete the RAG data entry itself from PostgreSQL
    db.delete(db_rag_data)
    
    db.commit()
    
    return {"message": f"RAG data entry '{rag_item_name}' (ID: {rag_id}) and all associated data have been completely deleted."}


@router.get("/{rag_id}/files", response_model=list[schemas.FileGist])
def list_rag_files(rag_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_active_user)):
    """Lists all files associated with a specific RAG data entry, based on ABAC policies."""
    rag_data = db.query(database.RagData).filter(database.RagData.id == rag_id).first()
    if rag_data is None:
        raise HTTPException(status_code=404, detail="RAG data not found")

    check_permission(db, current_user, "read_files", resource_type="rag_data", resource_id=rag_data.id)

    files = db.query(database.FileGist).filter(database.FileGist.rag_id == rag_id).all()
    # ... (rest of the URL generation logic remains the same)
    files_with_urls = []
    try:
        # Prepare clients for both first-party and third-party if needed
        default_minio_client = None
        if any(not f.is_third_party for f in files):
            if all([settings.MINIO_ENDPOINT, settings.MINIO_ACCESS_KEY, settings.MINIO_SECRET_KEY, settings.MINIO_BUCKET_NAME]):
                secure = "https" in settings.MINIO_ENDPOINT
                clean_endpoint = settings.MINIO_ENDPOINT.replace("https://", "").replace("http://", "")
                default_minio_client = Minio(clean_endpoint, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY, secure=secure)
        
        customer_minio_client = None
        if any(f.is_third_party for f in files):
            if all([settings.CUSTOMER_MINIO_ENDPOINT, settings.CUSTOMER_MINIO_ACCESS_KEY, settings.CUSTOMER_MINIO_SECRET_KEY, settings.CUSTOMER_MINIO_BUCKET_NAME]):
                secure = "https" in settings.CUSTOMER_MINIO_ENDPOINT
                clean_endpoint = settings.CUSTOMER_MINIO_ENDPOINT.replace("https://", "").replace("http://", "")
                customer_minio_client = Minio(clean_endpoint, access_key=settings.CUSTOMER_MINIO_ACCESS_KEY, secret_key=settings.CUSTOMER_MINIO_SECRET_KEY, secure=secure)

        for file_gist in files:
            file_gist_schema = schemas.FileGist.from_orm(file_gist)
            try:
                if file_gist.is_third_party:
                    if customer_minio_client:
                        bucket_name = settings.CUSTOMER_MINIO_BUCKET_NAME
                        client = customer_minio_client
                    else:
                        raise ValueError("Customer MinIO client not configured for a third-party file.")
                else:
                    if default_minio_client:
                        bucket_name = settings.MINIO_BUCKET_NAME
                        client = default_minio_client
                    else:
                        raise ValueError("Default MinIO client not configured for a first-party file.")
                
                download_url = client.presigned_get_object(bucket_name, file_gist.filename, expires=timedelta(days=7))
                file_gist_schema.download_url = download_url
            except (S3Error, ValueError) as e:
                print(f"MinIO Error generating URL for file '{file_gist.filename}': {e}")
                # Append schema without URL on error
            files_with_urls.append(file_gist_schema)
            
        return files_with_urls
    except Exception as e:
        print(f"An unexpected error occurred during MinIO URL generation: {e}")
        return [schemas.FileGist.from_orm(f) for f in files] # Return original list on major error


@router.delete("/files/{file_id}", status_code=status.HTTP_200_OK)
async def delete_rag_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
) -> Dict[str, str]:
    """
    Deletes a specific file and all its associated data across all systems
    (MinIO, Milvus, MongoDB, and the database record), based on ABAC policies.
    """
    # First, retrieve the file to get its associated RAG ID for the permission check.
    db_file = (db.query(database.FileGist).options(joinedload(database.FileGist.rag_data)).filter(database.FileGist.id == file_id).first())    
    if db_file is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    if not db_file.rag_data:
        # This case is unlikely if foreign keys are set up, but it's a good safeguard.
        # We'll attempt to delete the orphaned FileGist record and return.
        db.delete(db_file)
        db.commit()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated RAG item not found. Orphaned file record was cleaned up.")

    # Perform the permission check before proceeding with the deletion.
    check_permission(db, current_user, "delete_file", resource_type="rag_data", resource_id=db_file.rag_data.id)

    # Call the centralized purge service
    success = await purge_file_and_all_related_data(db, file_id)

    if success:
        return {"message": "File and all associated data deleted successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during the file deletion process. Check server logs for details.")


@router.get("/files/{file_id}/preview")
async def preview_rag_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
):
    """Retrieves file content or a pre-signed URL for preview, based on ABAC policies."""
    file_gist = db.query(FileGist).filter(FileGist.id == file_id).first()
    if not file_gist:
        raise HTTPException(status_code=404, detail="File not found")

    rag = db.query(database.RagData).filter(database.RagData.id == file_gist.rag_id).first()
    if not rag:
        raise HTTPException(status_code=404, detail="Associated RAG item not found")

    check_permission(db, current_user, "preview_file", resource_type="rag_data", resource_id=rag.id)
    logger.info(f"--- ROO DEBUG: Checking file preview for file_id: {file_id}, is_third_party: {file_gist.is_third_party} ---")
 
     # ... (rest of the preview logic remains the same)
    minio_client = None
    minio_bucket_name = None

    try:
        if file_gist.is_third_party:
            logger.info("--- ROO DEBUG: is_third_party is True, using CUSTOMER_MINIO settings. ---")
            endpoint = settings.CUSTOMER_MINIO_ENDPOINT
            access_key = settings.CUSTOMER_MINIO_ACCESS_KEY
            secret_key = settings.CUSTOMER_MINIO_SECRET_KEY
            minio_bucket_name = settings.CUSTOMER_MINIO_BUCKET_NAME
            logger.info(f"--- ROO DEBUG: CUSTOMER_MINIO_ENDPOINT: {endpoint} ---")
            logger.info(f"--- ROO DEBUG: CUSTOMER_MINIO_ACCESS_KEY: {access_key} ---")
            if not all([endpoint, access_key, secret_key, minio_bucket_name]):
                raise ValueError("CUSTOMER MinIO settings are not fully configured.")
            secure = "https" in endpoint
            clean_endpoint = endpoint.replace("https://", "").replace("http://", "")
            minio_client = Minio(clean_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        else:
            logger.info("--- ROO DEBUG: is_third_party is False, using default MINIO settings. ---")
            endpoint = settings.MINIO_ENDPOINT
            access_key = settings.MINIO_ACCESS_KEY
            secret_key = settings.MINIO_SECRET_KEY
            minio_bucket_name = settings.MINIO_BUCKET_NAME
            logger.info(f"--- ROO DEBUG: MINIO_ENDPOINT: {endpoint} ---")
            logger.info(f"--- ROO DEBUG: MINIO_ACCESS_KEY: {access_key} ---")
            if not all([endpoint, access_key, secret_key, minio_bucket_name]):
                raise ValueError("Default MinIO settings are not fully configured.")
            secure = "https" in endpoint
            clean_endpoint = endpoint.replace("https://", "").replace("http://", "")
            minio_client = Minio(clean_endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

        object_name = file_gist.filename
        file_extension = os.path.splitext(object_name)[1].lower()

        if file_extension == '.pdf':
            response = minio_client.get_object(minio_bucket_name, object_name)
            file_content = io.BytesIO(response.read())
            response.close()
            response.release_conn()
            return StreamingResponse(file_content, media_type='application/pdf', headers={"Content-Disposition": f"inline; filename=\"{os.path.basename(object_name.encode('utf-8').decode('latin-1'))}\""})
        else:
            url = minio_client.presigned_get_object(minio_bucket_name, object_name, expires=timedelta(days=7))
            file_type = 'text' if file_extension in ['.txt', '.md'] else 'image' if file_extension in ['.jpg', '.jpeg', '.png', '.gif'] else 'unknown'
            return {"url": url, "file_type": file_type}
            
    except (S3Error, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Error generating file preview: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post("/{rag_id}/uploadfile")
async def upload_rag_file(
    rag_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
) -> JSONResponse:
    """Uploads a file associated with a RAG data entry to MinIO, based on ABAC policies."""
    rag_entry = db.query(RagData).filter(RagData.id == rag_id).first()
    if not rag_entry:
        raise HTTPException(status_code=404, detail="RAG entry not found")

    check_permission(db, current_user, "upload_file", resource_type="rag_data", resource_id=rag_entry.id)

    # ... (rest of the upload logic remains the same)
    minio_bucket_name = settings.MINIO_BUCKET_NAME
    if not minio_bucket_name:
        raise HTTPException(status_code=500, detail="MINIO_BUCKET_NAME environment variable not set.")
    file_location = None
    try:
        # Use a with statement for the temporary file to ensure it's handled correctly
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file_content = await file.read()
            temp_file.write(file_content)
            file_location = temp_file.name

        # Calculate file hash
        import hashlib
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Check for duplicates in the same RAG entry
        existing_file = db.query(FileGist).filter(FileGist.rag_id == rag_id, FileGist.file_hash == file_hash).first()
        if existing_file:
            raise HTTPException(status_code=409, detail=f"This exact file already exists in this RAG entry (File ID: {existing_file.id}).")

        object_name = f"{rag_entry.name}/{file.filename}"
        
        # MinIO Operations
        minio_endpoint = settings.MINIO_ENDPOINT
        minio_access_key = settings.MINIO_ACCESS_KEY
        minio_secret_key = settings.MINIO_SECRET_KEY
        if not all([minio_endpoint, minio_access_key, minio_secret_key]):
             raise HTTPException(status_code=500, detail="MinIO credentials environment variables not set.")
        
        minio_client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)
        if not minio_client.bucket_exists(minio_bucket_name):
            minio_client.make_bucket(minio_bucket_name)
            
        minio_client.fput_object(minio_bucket_name, object_name, file_location)
        minio_url = f"/{minio_bucket_name}/{object_name}"

        # Create and commit the FileGist object with the hash
        db_gist = FileGist(
            filename=object_name,
            gist=f"MinIO Object: {object_name}",
            rag_id=rag_id,
            file_hash=file_hash
        )
        db.add(db_gist)
        db.commit()
        db.refresh(db_gist)
        
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded.", "file_gist_id": db_gist.id, "minio_url": minio_url})
    except Exception as e:
        import traceback
        print(f"!!! DETAILED UPLOAD ERROR: {str(e)}")
        print(f"!!! TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")
    finally:
        if file_location and os.path.exists(file_location):
            os.remove(file_location)


@router.post("/{rag_id}/process_document_for_preview", response_class=JSONResponse)
async def process_document_for_preview(
   rag_id: int,
   file: UploadFile = File(...),
   db: Session = Depends(get_db),
   current_user: User = Depends(auth.get_current_active_user)
):
   """
   Processes an uploaded document using the full pipeline (sniffing, MinerU/PaddleOCR)
   and returns the extracted plain text for preview purposes.
   This does NOT save the file or its embeddings.
   """
   rag_entry = db.query(RagData).filter(RagData.id == rag_id).first()
   if not rag_entry:
       raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="RAG entry not found")

   # Use the same permission as uploading a file
   check_permission(db, current_user, "upload_file", resource_type="rag_data", resource_id=rag_entry.id)

   try:
       # Call the improved function that handles the entire preview process
       processed_text = await get_text_from_uploaded_file(file)
       
       # 检查是否为错误信息
       is_error = processed_text and (processed_text.startswith("错误") or processed_text.startswith("无法"))
       
       return JSONResponse(
           status_code=status.HTTP_200_OK,
           content={
               "filename": file.filename, 
               "processed_text": processed_text,
               "success": not is_error,
               "file_size": file.size if hasattr(file, 'size') else len(await file.read()) if file else 0,
               "message": "文档处理成功" if not is_error else "文档处理遇到问题，请查看详细信息"
           }
       )
   except Exception as e:
       # Log the exception for debugging
       import traceback
       logger.error(f"文档预览处理发生异常: {e}")
       logger.error(f"详细错误信息: {traceback.format_exc()}")
       
       # 提供更友好的错误信息
       error_detail = f"文档处理失败。可能的原因：\n1. 文件格式不受支持或已损坏\n2. 文件过大或包含复杂内容\n3. 服务器处理能力不足\n\n技术详情: {str(e)}"
       
       raise HTTPException(
           status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
           detail=error_detail
       )


@router.post("/query")
async def query_rag_system(
    query: str = Query(..., description="The user's query for the RAG system."),
    rag_id: int = Query(..., description="The ID of the RAG entry to query."),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user)
):
    """
    Queries the RAG system using a hybrid search approach.
    """
    rag_item = db.query(database.RagData).filter(database.RagData.id == rag_id).first()
    if not rag_item:
        raise HTTPException(status_code=404, detail="RAG item not found")

    check_permission(db, current_user, "query", resource_type="rag_data", resource_id=rag_item.id)

    # ... (rest of the query analysis and Milvus search logic remains the same)
    llm = get_llm()
    prompt_template = '''...''' # Template remains the same
    prompt = prompt_template.format(user_query=query)
    try:
        llm_response = await llm.ainvoke(prompt)
        response_content = llm_response.content if hasattr(llm_response, 'content') else llm_response
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        parsed_response = json.loads(response_content)
        search_filter = parsed_response.get("filter", {})
        semantic_query = parsed_response.get("query", query)
        if not semantic_query:
            semantic_query = query
    except Exception as e:
        print(f"LLM response parsing failed: {e}. Falling back to simple search.")
        search_filter = {}
        semantic_query = query
    filter_expr = ""
    if search_filter and search_filter.get("filename"):
        filter_expr = f"original_filename == '{search_filter['filename']}'"
    try:
        sanitized_rag_item_name = rag_item.name.lower().replace(" ", "_")
        collection_name = f"rag_{sanitized_rag_item_name}"
        results = await search_in_milvus(query_text=semantic_query, collection_name=collection_name, filter_expr=filter_expr)
        return JSONResponse(content={"results": results})
    except Exception as e:
        print(f"Error searching in Milvus: {e}")
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")

