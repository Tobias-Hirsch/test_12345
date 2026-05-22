# `modules/milvus_module.py` - Milvus Kommentar

Hinweis`backend/app/modules/milvus_module.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentaröschen Milvus Kommentar
*   **Kommentar**: ErlaubenKommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar

## Kommentar
1.  **Milvus Hinweis**: Hinweis`pymilvus` Hinweis`connections.connect` Hinweis
    ```python
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    from backend.app.core.config import settings
    import logging

    logger = logging.getLogger(__name__)

    async def connect_to_milvus():
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            logger.info(f"Successfully connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            # Kommentar
            if not utility.has_collection(settings.MILVUS_COLLECTION_NAME):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.MILVUS_VECTOR_DIM),
                    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="chunk_id", dtype=DataType.INT64),
                    FieldSchema(name="chunk_content", dtype=DataType.TEXT),
                    FieldSchema(name="summary", dtype=DataType.TEXT),
                    FieldSchema(name="download_url", dtype=DataType.VARCHAR, max_length=1024)
                ]
                schema = CollectionSchema(fields, "Rosti RAG Collection for document embeddings")
                collection = Collection(settings.MILVUS_COLLECTION_NAME, schema)
                # Kommentar
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Collection '{settings.MILVUS_COLLECTION_NAME}' created and indexed successfully.")
            else:
                logger.info(f"Collection '{settings.MILVUS_COLLECTION_NAME}' already exists.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus or create collection: {e}")
            raise

    def get_milvus_collection() -> Collection:
        if not utility.has_collection(settings.MILVUS_COLLECTION_NAME):
            raise Exception(f"Milvus collection '{settings.MILVUS_COLLECTION_NAME}' does not exist.")
        collection = Collection(settings.MILVUS_COLLECTION_NAME)
        collection.load() # Hinweis
        return collection

    def release_milvus_collection():
        if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
            collection = Collection(settings.MILVUS_COLLECTION_NAME)
            collection.release()
            logger.info(f"Collection '{settings.MILVUS_COLLECTION_NAME}' released.")
    ```
2.  **Hinweis**: `Collection.insert` Hinweis
3.  **Hinweis**: `Collection.search` Hinweisührt ausHinweis
4.  **Hinweis**: Hinweis`collection.create_index` Hinweis

## Kommentar
`/backend/app/modules/milvus_module.py`