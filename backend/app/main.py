# Developer: Jinglu Han
# mailbox: admin@de-manufacturing.cn

from dotenv import load_dotenv
load_dotenv(override=True)

import logging
from fastapi import FastAPI
import os
from app.core.config import settings # Ensure settings are loaded early
from app.rag_knowledge.generic_knowledge import connect_to_milvus
from app.models.database import get_db
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.services.inactive_user_cleaner import remove_expired_unactivated_users
from app.services.conversation_cleaner import remove_old_conversations
from app.initial_data import initialize_data
from app.services.minio_sync_service import sync_minio_bucket # Import the new sync service

# Import routers
from app.routers import captcha
from app.routers import authentication
from app.routers import smtp
from app.routers import rag
from app.routers import embeddings
#from app.routers import knowledge_base # Import the new router
from app.routers import roles
from app.routers import user_roles
from app.routers import users
from app.routers import settings as settings_router
from app.routers import chat
from app.routers import files
from app.routers import policies
from app.routers import agent_chat
from app.routers import abac

# Create scheduler instance globally
scheduler = AsyncIOScheduler(timezone=settings.TIMEZONE)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler for startup and shutdown tasks.
    """
    # Startup events
    logging.basicConfig(level=logging.DEBUG) # Configure logging here
    logger.info("Application startup: Creating database tables and connecting to Milvus.")
    # The following line is removed as database creation and migration are now handled by Alembic.
    # create_database_tables()
    
    # Initialize roles, policies, and admin user
    with next(get_db()) as db:
        await initialize_data(db)
    
    await connect_to_milvus() # Keep Milvus connection logic
    
    logger.info("Application startup: Starting scheduler and adding cleanup jobs.")
    scheduler.start()
    scheduler.add_job(remove_expired_unactivated_users, 'interval', days=1, id='user_cleanup_job')
    scheduler.add_job(remove_old_conversations, 'interval', days=1, id='conversation_cleanup_job')
    
    # Configure and add the new MinIO sync job if it's enabled
    if settings.MINIO_SYNC_ENABLED:
        logger.info("MinIO sync is enabled. Scheduling job...")
        start_hour = int(settings.MINIO_SYNC_CRON_START_HOUR)
        end_hour = int(settings.MINIO_SYNC_CRON_END_HOUR)
        minute = settings.MINIO_SYNC_CRON_MINUTE
        
        # Handle overnight cron scheduling (e.g., 22:00 to 06:00)
        if start_hour > end_hour:
            hour_cron = f"{start_hour}-23,0-{end_hour}"
        else:
            hour_cron = f"{start_hour}-{end_hour}"
            
        logger.info(f"Scheduling MinIO sync job with cron: hour='{hour_cron}', minute='{minute}'")
        scheduler.add_job(
            sync_minio_bucket,
            'cron',
            hour=hour_cron,
            minute=minute,
            id='minio_sync_job'
        )
    else:
        logger.info("MinIO sync is disabled. Skipping job scheduling.")
    
    print(f"MILVUS_HOST from .env after load_dotenv: {settings.MILVUS_HOST}") # Add print statement for debugging
    
    yield
    
    # Shutdown events
    logger.info("Application shutdown: Shutting down scheduler.")
    scheduler.shutdown()
    print("Scheduler shut down.")

app = FastAPI(lifespan=lifespan) # Pass the lifespan context manager

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware # Import SessionMiddleware

origins = [
    "http://localhost",
    "http://localhost:8080", # Frontend URL
    "http://127.0.0.1:8080", # Frontend URL
    "http://localhost:5173", # Common Vue dev server port
    "http://127.0.0.1:5173", # Common Vue dev server port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins to resolve CORS issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Session Middleware for OAuth
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# Include routers
app.include_router(captcha.router)
app.include_router(authentication.router)
app.include_router(smtp.router)
app.include_router(rag.router, prefix="/rag")
app.include_router(embeddings.router, prefix="/rag")
#app.include_router(knowledge_base.router, prefix="/rag") # Include the new router
app.include_router(roles.router)
app.include_router(user_roles.router)
app.include_router(users.router)
app.include_router(settings_router.router)
app.include_router(chat.router, prefix="/chat")
app.include_router(policies.router)
app.include_router(files.router, prefix="/files")
app.include_router(agent_chat.router)
app.include_router(abac.router)


@app.get("/")
async def read_root():
    return {"message": "Backend service is running!"}


# @app.get("/debug-routes")
# async def debug_routes():
#     """Returns a list of all registered routes in the FastAPI application."""
#     routes_list = []
#     for route in app.routes:
#         if hasattr(route, "path") and hasattr(route, "methods"):
#             routes_list.append({
#                 "path": route.path,
#                 "methods": list(route.methods) if route.methods else [],
#                 "name": route.name,
#             })
#     return {"routes": routes_list}







