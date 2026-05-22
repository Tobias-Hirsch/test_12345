# `modules/mongodb_module.py` - MongoDB Kommentar

Hinweis`backend/app/modules/mongodb_module.py` Hinweis

## Kommentar
*   **Kommentar**: Kommentar
*   **DokumenteAktionen**: KommentaröschenDokumente. 
*   **Kommentar**: Kommentar

## Kommentar
1.  **MongoDB Hinweis**: Hinweis`pymongo` Hinweis`MongoClient` Hinweis
    ```python
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    from backend.app.core.config import settings
    import logging

    logger = logging.getLogger(__name__)

    client: MongoClient = None

    def connect_to_mongodb():
        global client
        try:
            client = MongoClient(settings.MONGODB_URI)
            # The ping command is cheap and does not require auth.
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during MongoDB connection: {e}")
            raise

    def get_mongodb_client() -> MongoClient:
        if client is None:
            raise Exception("MongoDB client not initialized. Call connect_to_mongodb() first.")
        return client

    def get_database():
        return get_mongodb_client()[settings.MONGODB_DATABASE_NAME]

    def get_collection(collection_name: str):
        return get_database()[collection_name]
    ```
2.  **Hinweis**: Hinweis
3.  **CRUD Aktionen**: Hinweis`get_collection` Hinweis
    *   `insert_one(document)`
    *   `find_one(query)`
    *   `find(query)`
    *   `update_one(query, update)`
    *   `delete_one(query)`

## Kommentar
`/backend/app/modules/mongodb_module.py`