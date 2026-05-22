# Developer: Jinglu Han
# mailbox: admin@de-manufacturing.cn

# backend/app/services/chat_response_service.py
import os
import uuid
import shutil
import logging
import asyncio
import tempfile
import traceback
import re
from typing import Dict, Any, AsyncGenerator, List
from datetime import datetime
import tiktoken

from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.database import User, RagData
from app.schemas import chat_schemas
from app.services import chat_data_service, ollama_service
from app.services.conversation_service import ConversationService
from app.rag_knowledge.generic_knowledge import retrieve_rag_context
from app.tools.search_online_tools import duckduckgosearch
from app.tools.document_processor import summarize_long_text
from app.tools.deal_document import extract_text_from_file_content
from app.services.rag_permission_service import RagPermissionService
from app.core.config import settings
from app.modules.minio_module import get_document_from_minio, get_document_bytes_from_minio

logger = logging.getLogger(__name__)

# System prompt for direct AI chat mode, encouraging detailed responses.
DIRECT_CHAT_SYSTEM_PROMPT = 'Du bist ein hilfreicher Assistent. Antworte umfassend, strukturiert und mit nachvollziehbaren Erklärungen. Deine Antwort soll grundsätzlich ausführlich sein und, sofern die Anfrage es zulässt, mindestens 1000 Wörter umfassen.'

class ChatResponseService:
    def __init__(self, db: Session, redis_client, conversation_service: ConversationService):
        self.db = db
        self.redis_client = redis_client
        self.conversation_service = conversation_service

    async def generate_response(
        self,
        conversation_id: str,
        message_create: chat_schemas.MessageCreate,
        current_user: User,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main entry point to generate a bot response. It prepares the state,
        gathers context, and streams the final LLM answer.
        """
        # 1. Get or initialize conversation state
        conversation_state = self._get_or_initialize_conversation_state(current_user, conversation_id)

        # 2. Add user message to state
        self._add_user_message_to_state(conversation_state, message_create, current_user, conversation_id)

        # 3. Stream the response generation process
        return self._master_stream_generator(conversation_id, message_create, current_user, conversation_state)

    def _get_or_initialize_conversation_state(self, current_user: User, conversation_id: str) -> Dict[str, Any]:
        """Gets conversation state from Redis or initializes it from DB."""
        conversation_state = self.conversation_service.get_conversation_state(current_user.id, conversation_id)
        if not conversation_state:
            conversation_from_db = asyncio.run(chat_data_service.get_conversation_by_id(conversation_id))
            if not conversation_from_db:
                raise HTTPException(status_code=404, detail="Conversation not found")
            if conversation_from_db['user_id'] != str(current_user.id):
                raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
            
            conversation_state = {
                "history": conversation_from_db.get("messages", []),
                "context": {}, "variables": {}, "attached_files": []
            }
            self.conversation_service.save_conversation_state(current_user.id, conversation_id, conversation_state)
        return conversation_state

    def _add_user_message_to_state(self, state: Dict, message: chat_schemas.MessageCreate, user: User, conv_id: str):
        """Adds the user's message to the conversation history in state."""
        user_message_entry = {
            "sender": "user",
            "content": message.content,
            "attachments": [att.model_dump() for att in message.attachments] if message.attachments else [],
            "timestamp": datetime.now().isoformat(),
            "_id": str(uuid.uuid4())
        }
        state["history"].append(user_message_entry)
        
        if message.attachments:
            for att in message.attachments:
                state["attached_files"].append(att.model_dump())
        
        self.conversation_service.save_conversation_state(user.id, conv_id, state)

    async def _master_stream_generator(
        self,
        conversation_id: str,
        message_create: chat_schemas.MessageCreate,
        current_user: User,
        conversation_state: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Orchestrates the entire response generation process as a stream."""
        bot_message_id = str(uuid.uuid4())
        self._add_loading_message_to_state(current_user.id, conversation_id, bot_message_id)
        if message_create.show_think_process:
            yield {"event": "thought", "data": "Thinking..."}

        # --- Context Gathering ---
        rag_context, rag_sources = "", []
        online_context, online_sources = "", []
        attachment_context = ""

        tasks = []
        if message_create.search_rosti_active:
            yield {"event": "thought", "data": "\n- Searching Rosti Data..."}
            # Create a cleaned version of the query for RAG search
            cleaned_query_for_rag = self._clean_query(message_create.content)
            tasks.append(asyncio.create_task(self._run_rag_search_task(cleaned_query_for_rag, current_user, message_create.show_think_process)))
        
        if message_create.search_online_active:
            yield {"event": "thought", "data": "\n- Searching online..."}
            tasks.append(asyncio.create_task(self._run_online_search_task(message_create.content)))

        if message_create.attachments:
            yield {"event": "thought", "data": "\n- Processing attachments..."}
            async for chunk in self._process_attachments_stream(message_create, message_create.show_think_process):
                # Check for the internal event to capture the final content
                if chunk.get("event") == "internal_content":
                    attachment_context = chunk["data"]
                else:
                    # Pass through all other valid events (like 'thought' from summarization)
                    yield chunk

        # Await concurrent search tasks
        if tasks:
            yield {"event": "thought", "data": "\n\nGathering all information..."}
            results = await asyncio.gather(*tasks)
            # Unpack results based on which tasks were run
            task_idx = 0
            if message_create.search_rosti_active:
                rag_context, rag_sources = results[task_idx]
                task_idx += 1
            if message_create.search_online_active:
                online_context, online_sources = results[task_idx]

        # --- Final Response Generation ---
        yield {"event": "thought", "data": "\nSynthesizing final answer..."}
        
        # DEBUG: Log context information
        logger.info(f"=== FINAL CONTEXT DEBUG ===")
        logger.info(f"rag_context length: {len(rag_context)}")
        logger.info(f"rag_context preview: {rag_context[:300] if rag_context else 'Empty'}...")
        logger.info(f"attachment_context length: {len(attachment_context)}")
        logger.info(f"attachment_context preview: {attachment_context[:300] if attachment_context else 'Empty'}...")
        logger.info(f"=== END CONTEXT DEBUG ===")
        
        search_results_context = f"\n\nRelevant Rosti Data:\n{rag_context}" if rag_context else ""
        search_results_context += online_context
        search_results_context += attachment_context
        
        source_documents = self._deduplicate_sources(rag_sources + online_sources)

        # --- Dual-mode prompt logic ---
        # Keep the user's visible question clean. Backend options decide whether
        # thought output is shown; model-control tokens should not leak into the
        # semantic prompt, especially for file/RAG questions.
        final_user_query_for_llm = message_create.content

        # If RAG or Online Search is active, use the structured prompt
        if message_create.search_rosti_active or message_create.search_online_active or message_create.attachments:
            rag_prompt = self._construct_rag_system_prompt(final_user_query_for_llm, search_results_context, conversation_state, message_create)
            llm_messages = [
                {
                    'role': 'system',
                    'content': 'Du bist ein fachkundiger KI-Assistent. Beantworte die Anfrage anhand der bereitgestellten Kontextinformationen.'
                },
                {'role': 'user', 'content': rag_prompt}
            ]
        else:
            # "Direct Chat" mode: just use the history and a simple system message
            # Restore history and use a simple prompt. Temperature will be added later.
            history_messages = self._get_token_limited_history(conversation_state, settings.MAX_CONTEXT_TOKENS)
            system_prompt_content = DIRECT_CHAT_SYSTEM_PROMPT
            llm_messages = [{"role": "system", "content": system_prompt_content}] + history_messages + [{'role': 'user', 'content': final_user_query_for_llm}]

        full_bot_response_content = ""
        try:
            # Determine temperature based on context
            is_rag_mode = message_create.search_rosti_active or message_create.search_online_active or message_create.attachments
            temperature = 0.3 if is_rag_mode else 0.8
            
            # Ollama option names are not identical to OpenAI option names.
            # Use num_predict instead of max_tokens and avoid unsupported penalty fields.
            llm_options = {
                "num_predict": settings.OLLAMA_NUM_PREDICT,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
            }
            async for chunk in ollama_service.get_chat_response_stream(
                llm_messages,
                show_think_process=message_create.show_think_process,
                options=llm_options
            ):
                full_bot_response_content += chunk
                yield {"event": "text", "data": chunk}

            logger.debug(f"--- DEBUG: Final source_documents to be yielded: {source_documents}")
            yield {"event": "metadata", "data": {"source_documents": source_documents}}
            
            self._save_final_bot_message(current_user.id, conversation_id, bot_message_id, full_bot_response_content, source_documents)

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error during final LLM streaming or saving: {e}\nTraceback: {error_traceback}")
            yield {"event": "error", "data": f"An error occurred during final response generation: {e}"}

    def _add_loading_message_to_state(self, user_id: int, conv_id: str, bot_message_id: str):
        """Adds a temporary 'bot is typing' message to Redis."""
        state = self.conversation_service.get_conversation_state(user_id, conv_id)
        loading_message = {
            "_id": bot_message_id, "sender": "bot", "content": "", "loading": True,
            "attachments": [], "source_documents": [], "timestamp": datetime.now().isoformat()
        }
        state["history"].append(loading_message)
        self.conversation_service.save_conversation_state(user_id, conv_id, state)

    async def _run_rag_search_task(self, query_text: str, current_user: User, show_think_process: bool):
        """Performs RAG search using a cleaned query."""
        try:
            # Pass both the db session and the redis client to the permission service
            permission_service = RagPermissionService(self.db, self.redis_client)
            permitted_rag_items = permission_service.get_accessible_rags(current_user)
            if not permitted_rag_items:
                return "No relevant Rosti Data found (due to permissions).", []

            return await retrieve_rag_context(
                query_text=query_text, db_session=self.db,
                permitted_rag_items=permitted_rag_items
            )
        except Exception as e:
            logger.error(f"Error during RAG search task: {e}", exc_info=True)
            return f"\n\nError during RAG search: {e}", []

    async def _run_online_search_task(self, query: str):
        """Performs online search."""
        try:
            results_dict = await duckduckgosearch([query], max_results=5)
            results = results_dict.get(query, [])
            if results:
                context = "\n\nRelevant Online Search Results:\n" + "\n".join([f"Title: {o.get('title', 'N/A')}\nSnippet: {o.get('body', 'N/A')}" for o in results])
                sources = [{"type": "Online", "title": o.get('title', 'N/A'), "snippet": o.get('body', '')[:100] + '...'} for o in results]
                return context, sources
            return "No relevant online results found.", []
        except Exception as e:
            logger.error(f"Error during Online search: {e}")
            return f"\n\nError during Online search: {e}", []

    async def _process_attachments_stream(self, message_create: chat_schemas.MessageCreate, show_think_process: bool):
        """
        Processes multiple file attachments using the unified document processing service,
        and streams summarization if needed.
        """
        if not message_create.attachments:
            return

        full_extracted_content = []
        total_attachments = len(message_create.attachments)

        for i, att in enumerate(message_create.attachments):
            try:
                if show_think_process:
                    yield {"event": "thought", "data": f"\n- Processing attachment {i+1}/{total_attachments}: {att.filename}..."}
                
                # Get file bytes directly from MinIO
                file_bytes = await get_document_bytes_from_minio(att.object_name, att.bucket_name)
                
                # Use the unified document processing service, passing the user's query for context (especially for images)
                content = await extract_text_from_file_content(
                    file_bytes,
                    att.filename,
                    question=message_create.content
                )
                
                # DEBUG: Log the extracted content details
                logger.info(f"=== ATTACHMENT DEBUG: {att.filename} ===")
                logger.info(f"Content type: {type(content)}")
                logger.info(f"Content length: {len(content) if content else 0}")
                logger.info(f"Content preview: {content[:200] if content else 'None'}...")
                logger.info(f"=== END ATTACHMENT DEBUG ===")
                
                if content:
                    full_extracted_content.append(f"\n\n--- Content from attachment: {att.filename} ---\n{content}")
                else:
                    logger.warning(f"No content extracted from attachment: {att.filename}")
                    full_extracted_content.append(f"\n\n--- Content from attachment: {att.filename} ---\n[No content could be extracted from this file]")

            except Exception as e:
                error_message = f"\n\nError processing attachment '{att.filename}': {e}"
                logger.error(f"Error processing attachment {att.object_name}: {e}", exc_info=True)
                full_extracted_content.append(error_message)
                if show_think_process:
                    yield {"event": "thought", "data": f"\n- Failed to process {att.filename}."}

        # Combine content from all attachments
        combined_content = "\n".join(full_extracted_content)
        
        # DEBUG: Log the final combined content
        logger.info(f"=== COMBINED ATTACHMENT CONTENT DEBUG ===")
        logger.info(f"Total attachments processed: {len(full_extracted_content)}")
        logger.info(f"Combined content length: {len(combined_content)}")
        logger.info(f"Combined content preview: {combined_content[:500] if combined_content else 'None'}...")
        logger.info(f"=== END COMBINED CONTENT DEBUG ===")

        # Check if user wants translation (don't summarize for translation tasks)
        user_query_lower = message_create.content.lower()
        is_translation_task = any(keyword in user_query_lower for keyword in ['hinweis', 'translate', 'übersetze', 'uebersetze', 'deutsch', 'chinesisch', 'chinese'])
        
        # Summarize if the combined content is too long AND it's not a translation task
        if len(combined_content) > settings.LONG_TEXT_THRESHOLD and not is_translation_task:
            if show_think_process:
                yield {"event": "thought", "data": "\n- Combined content is long, starting summarization...\n"}
            
            # Send periodic heartbeat to prevent connection timeout
            import time
            last_heartbeat = time.time()
            
            summary_stream = summarize_long_text(combined_content, show_think_process=show_think_process)
            final_summary = ""
            chunk_count = 0
            
            async for chunk in summary_stream:
                chunk_count += 1
                current_time = time.time()
                
                # Send heartbeat every 10 seconds to prevent connection timeout
                if current_time - last_heartbeat > 10:
                    yield {"event": "heartbeat", "data": f"Processing... ({chunk_count} chunks processed)"}
                    last_heartbeat = current_time
                
                if chunk.startswith("__FINAL_SUMMARY_COMPLETE__"):
                    final_summary = chunk.replace("__FINAL_SUMMARY_COMPLETE__:", "")
                    break
                if show_think_process:
                    yield {"event": "thought", "data": chunk}
                else:
                    # Even without show_think_process, send periodic progress updates
                    if chunk_count % 3 == 0:  # Every 3rd chunk
                        yield {"event": "progress", "data": f"Summarizing document... ({chunk_count} sections processed)"}
            
            yield {"event": "internal_content", "data": final_summary}
        else:
            # For translation tasks or short content, use original content
            if is_translation_task and len(combined_content) > settings.LONG_TEXT_THRESHOLD:
                if show_think_process:
                    yield {"event": "thought", "data": "\n- Translation task detected, preserving full original content...\n"}
            yield {"event": "internal_content", "data": combined_content}

    def _clean_query(self, query: str) -> str:
        """Removes command-like arguments (e.g., /nothink) from the query."""
        return re.sub(r'\s*/\w+', '', query).strip()

    def _get_token_limited_history(self, state: Dict, max_tokens: int) -> List[Dict[str, str]]:
        """
        Truncates conversation history to fit within a specified token limit.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = None

        history_for_prompt = []
        current_history_tokens = 0
        
        # Iterate backwards through history (from newest to oldest)
        for msg in reversed(state["history"][:-1]): # Exclude the latest user message
            # Ensure message format is correct for passing to LLM
            formatted_msg = {"role": msg.get("sender", "user"), "content": msg.get("content", "")}
            
            msg_tokens = len(encoding.encode(formatted_msg["content"])) if encoding else len(formatted_msg["content"]) / 3
            
            if current_history_tokens + msg_tokens <= max_tokens:
                current_history_tokens += msg_tokens
                history_for_prompt.insert(0, formatted_msg)
            else:
                break
        
        return history_for_prompt

    def _construct_rag_system_prompt(self, latest_user_query: str, context: str, state: Dict, message_create: chat_schemas.MessageCreate) -> str:
        """Builds the structured system prompt for RAG or Online Search modes."""
        
        fixed_context_tokens = len(latest_user_query + context) / 3
        history_token_budget = settings.MAX_CONTEXT_TOKENS - fixed_context_tokens - 1000
        history_messages = self._get_token_limited_history(state, history_token_budget)
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_messages])

        # --- Dynamically build instructions based on context and settings ---
        instructions = [
            "1. Konzentriere dich AUSSCHLIESSLICH auf die \"AKTUELLE FRAGE\".",
            "3. Nutze den \"GESPRÄCHSVERLAUF\" nur für Hintergrundinformationen, wenn er direkt zur aktuellen Frage gehört.",
            "4. Gib eine umfassende, relevante und gut formatierte Markdown-Antwort."
        ]

        context_is_empty = not context.strip() or "No real-time information was searched for or found" in context
        is_rag_search_mode = message_create.search_rosti_active or message_create.search_online_active or message_create.attachments

        if is_rag_search_mode:
            if context_is_empty:
                if not message_create.search_ai_active:
                    # Strict RAG mode, no context, no AI fallback -> Must stop.
                    instructions.insert(1, "2. Die Kontextinformationen sind leer. Du MUSST ausschließlich antworten: 'Es konnten keine relevanten Inhalte gefunden werden. Bitte passen Sie Ihre Suchbegriffe oder Ihre Frage an.' Beantworte die Frage auf keine andere Weise.")
                else:
                    # RAG mode with AI fallback, no context -> Announce and use general knowledge.
                    instructions.insert(1, "2. Du MUSST zuerst schreiben: 'In den bereitgestellten Unterlagen konnte keine direkte Antwort gefunden werden. Die folgende Antwort basiert auf meinem allgemeinen Wissen:' und anschließend mit allgemeinem Wissen antworten.")
            else:
                # RAG mode with context -> Must use context.
                instructions.append("6. Wenn der Benutzer allgemein auf eine Datei, Anlage, Quelle oder ein Dokument verweist, behandle die \"KONTEXTINFORMATIONEN\" als die bereitgestellten Dokumentinhalte.")
                instructions.append("7. Wenn der Benutzer um Analyse oder Zusammenfassung bittet, fasse die vorhandenen Dokumentinhalte strukturiert zusammen, statt die Anfrage wegen fehlender Details abzulehnen.")
                instructions.insert(1, "2. Stütze deine Antwort STRENG UND AUSSCHLIESSLICH auf die \"KONTEXTINFORMATIONEN\".")
                if not message_create.search_ai_active:
                    # Strict RAG, has context, but answer might not be in it.
                    instructions.append("5. Wenn die Antwort nicht in den \"KONTEXTINFORMATIONEN\" enthalten ist, MUSST du mitteilen, dass du die Antwort in den bereitgestellten Dokumenten nicht finden konntest. Nutze KEIN allgemeines Wissen.")
                else:
                    # RAG with AI fallback, has context, but can use general knowledge if needed.
                    instructions.append("5. Wenn die Antwort nicht in den \"KONTEXTINFORMATIONEN\" enthalten ist, darfst du allgemeines Wissen ergänzen. Du MUSST aber klar sagen, dass die primäre Information nicht in den Dokumenten gefunden wurde.")
        else:
            # This case should ideally not use this prompt function, but as a fallback:
            instructions.insert(1, "2. Nutze dein allgemeines Wissen und deine Schlussfolgerungen, um die Frage des Benutzers direkt zu beantworten.")

        instructions_str = "\n            ".join(instructions)
        # --- End of dynamic instructions ---

        return f"""Du bist ein fachkundiger KI-Assistent. Dein Hauptziel ist es, die AKTUELLE FRAGE des Benutzers direkt und präzise zu beantworten.

            **AKTUELLE FRAGE:**
            "{latest_user_query}"

            **KONTEXTINFORMATIONEN (zur Beantwortung der Frage nutzen):**
            {context if context.strip() else "Für diese Anfrage wurden keine Echtzeitinformationen gesucht oder gefunden."}

            **GESPRÄCHSVERLAUF (nur als Referenz, ignorieren, wenn nicht relevant):**
            {history_context if history_context.strip() else "Kein vorheriger Gesprächsverlauf."}

            ---
            ANWEISUNGEN:
            {instructions_str}
            """

    def _deduplicate_sources(self, sources: list) -> list:
        """Removes duplicate source documents based on filename."""
        unique_sources, seen_filenames = [], set()
        for source in sources:
            if isinstance(source, dict):
                filename = source.get("filename")
                if filename and filename not in seen_filenames:
                    unique_sources.append(source)
                    seen_filenames.add(filename)
                elif not filename: # For sources without a filename (e.g., online)
                    unique_sources.append(source)
            else:
                unique_sources.append(source)
        return unique_sources

    def _save_final_bot_message(self, user_id: int, conv_id: str, msg_id: str, content: str, sources: list):
        """Updates the bot's message in Redis with the final content."""
        final_state = self.conversation_service.get_conversation_state(user_id, conv_id)
        message_to_update = next((msg for msg in final_state["history"] if msg.get("_id") == msg_id), None)
        if message_to_update:
            message_to_update["content"] = content
            message_to_update["loading"] = False
            message_to_update["source_documents"] = sources
            self.conversation_service.save_conversation_state(user_id, conv_id, final_state)
            logger.info(f"Final bot response saved to Redis for conversation {conv_id}")
        else:
            logger.error(f"Could not find message {msg_id} to update after final response.")
