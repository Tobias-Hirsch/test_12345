# `modules/minio_module.py` - MinIO Kommentar

Hinweis`backend/app/modules/minio_module.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentaröschen**: KommentaröschenKommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
1.  **MinIO Hinweis**: Hinweis`minio` Hinweis`Minio` Hinweis
    ```python
    from minio import Minio
    from minio.error import S3Error
    from backend.app.core.config import settings
    import os

    minio_client: Minio = None

    async def connect_to_minio():
        global minio_client
        try:
            minio_client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=False # Hinweis
            )
            # Kommentar
            found = minio_client.bucket_exists(settings.MINIO_BUCKET_NAME)
            if not found:
                minio_client.make_bucket(settings.MINIO_BUCKET_NAME)
                print(f"Bucket '{settings.MINIO_BUCKET_NAME}' created successfully.")
            else:
                print(f"Bucket '{settings.MINIO_BUCKET_NAME}' already exists.")
        except S3Error as e:
            print(f"Error connecting to MinIO: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during MinIO connection: {e}")
            raise
    ```
2.  **Hinweis**: Hinweisührt ausHinweisöschenAktionen. 
    *   **`upload_file_to_minio(file_path: str, file_content: bytes)`**: Kommentar
    *   **`download_file_from_minio(file_path: str)`**: Kommentar
    *   **`delete_file_from_minio(file_path: str)`**: LöschenKommentar
3.  **Hinweis**: `presigned_get_object` Hinweis
    ```python
    def get_presigned_download_url(object_name: str, expires_in_seconds: int = 3600) -> str:
        if not minio_client:
            raise Exception("MinIO client not initialized.")
        try:
            url = minio_client.presigned_get_object(
                settings.MINIO_BUCKET_NAME,
                object_name,
                expires=timedelta(seconds=expires_in_seconds)
            )
            return url
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            raise
    ```

## Kommentar
`/backend/app/modules/minio_module.py`