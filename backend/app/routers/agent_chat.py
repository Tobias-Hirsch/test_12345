import io
import json
import logging
import os
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agents.code_execution_agent import CodeExecutionAgent
from app.agents.code_generation_agent import CodeGenerationAgent
from app.agents.file_processing_agent import FileProcessingAgent
from app.agents.formatting_output_agent import FormattingOutputAgent
from app.agents.information_retrieval_agent import InformationRetrievalAgent
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.reasoning_planning_agent import ReasoningPlanningAgent
from app.agents.search_agent import SearchAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.workflow_control_agent import WorkflowControlAgent
from app.models.database import get_db
from app.schemas import schemas
from app.services import auth
from app.services.rag_permission_service import RagPermissionService
from app.core.redis_client import get_redis_client
from app.utils.stream_processors import sse_stream_formatter # Import the new formatter

router = APIRouter()

logger = logging.getLogger(__name__)

# Initialize agents
orchestrator_agent = OrchestratorAgent()
reasoning_planning_agent = ReasoningPlanningAgent()
validation_agent = ValidationAgent()
information_retrieval_agent = InformationRetrievalAgent()
file_processing_agent = FileProcessingAgent()
search_agent = SearchAgent()
formatting_output_agent = FormattingOutputAgent()
code_generation_agent = CodeGenerationAgent()
code_execution_agent = CodeExecutionAgent()
workflow_control_agent = WorkflowControlAgent()

# Register agents with the orchestrator
orchestrator_agent.register_agent("Reasoning/Planning Agent", reasoning_planning_agent)
orchestrator_agent.register_agent("Validation Agent", validation_agent)
orchestrator_agent.register_agent("Information Retrieval Agent", information_retrieval_agent)
orchestrator_agent.register_agent("Conversation Attachment File Processing Agent", file_processing_agent)
orchestrator_agent.register_agent("Search Agent", search_agent)
orchestrator_agent.register_agent("Formatting/Output Agent", formatting_output_agent)
orchestrator_agent.register_agent("Code Generation Agent", code_generation_agent)
orchestrator_agent.register_agent("Code Execution Agent", code_execution_agent)
orchestrator_agent.register_agent("Workflow Control Agent", workflow_control_agent)


@router.post("/api/agent_chat")
async def agent_chat_endpoint(
    message: str = Form(...),
    display_thoughts: Optional[str] = Form(None),
    search_ai_active: Optional[str] = Form(None),
    search_rag_active: Optional[str] = Form(None),
    search_online_active: Optional[str] = Form(None),
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db),
    redis_client = Depends(get_redis_client)
):
    """
    Endpoint for interacting with the multi-agent system.
    Accepts text message and optional file uploads.
    """
    logger.info(f"Received message: {message}")

    if files is None:
        files_list = []
    elif not isinstance(files, list):
        files_list = [files]
    else:
        files_list = files

    processed_files_for_agents = []
    for uploaded_file in files_list:
        file_content = await uploaded_file.read()
        file_stream = io.BytesIO(file_content)
        
        processed_files_for_agents.append({
            "filename": uploaded_file.filename,
            "content_type": uploaded_file.content_type,
            "file_stream": file_stream
        })
        logger.info(f"Created BytesIO for {uploaded_file.filename}. Size: {file_stream.getbuffer().nbytes} bytes.")
        await uploaded_file.close()

    logger.info(f"Passing {len(processed_files_for_agents)} processed files to context.")

    # Get retrievable RAG IDs based on user permissions
    permission_service = RagPermissionService(db, redis_client)
    retrievable_rag_ids = permission_service.get_retrievable_rag_ids(current_user)
    logger.info(f"User '{current_user.username}' has access to RAG IDs: {retrievable_rag_ids}")

    context = {
        "user_id": current_user.id,
        "user_name": current_user.username,
        "display_thoughts": display_thoughts == 'true',
        "search_ai_active": search_ai_active == 'true',
        "search_rag_active": search_rag_active == 'true',
        "search_online_active": search_online_active == 'true',
        "db_session": db,
        "current_user": current_user,
        "files": processed_files_for_agents,
        "retrievable_rag_ids": retrievable_rag_ids  # Add RAG IDs to context
    }
    
    # The response generation is now handled by the centralized sse_stream_formatter
    response_generator = sse_stream_formatter(orchestrator_agent.process(message, context))
    return StreamingResponse(response_generator, media_type="text/event-stream")
