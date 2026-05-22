from typing import Any, Dict, Optional, AsyncGenerator
from .base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    """
    The Orchestrator Agent is the entry point for user requests.
    It understands the user's intent and delegates tasks to appropriate
    specialized agents. It also manages the overall workflow, including
    retry logic for the Validation Agent and coordinating responses from
    various agents.
    """

    def __init__(self):
        super().__init__(name="Orchestrator Agent",
                         description="Delegates tasks and manages overall workflow.")
        self.agents = {} # To store references to other agents

    def register_agent(self, agent_name: str, agent_instance: BaseAgent):
        """Registers a specialized agent with the orchestrator."""
        self.agents[agent_name] = agent_instance

    async def process(self, task: str, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a user task by delegating to appropriate agents and streaming the response.

        Args:
            task (str): The user's initial request or task.
            context (Dict[str, Any]): Initial context from the user, including db_session and current_user.

        Yields:
            Dict[str, Any]: Chunks of the final result from the multi-agent system.
        """
        thoughts = []
        db_session = context.get("db_session")
        current_user = context.get("current_user")

        def clean_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
            """Removes non-JSON serializable objects from the chunk."""
            cleaned_chunk = chunk.copy()
            if "db_session" in cleaned_chunk:
                del cleaned_chunk["db_session"]
            if "current_user" in cleaned_chunk:
                del cleaned_chunk["current_user"]
            
            if "agent_output" in cleaned_chunk and isinstance(cleaned_chunk["agent_output"], dict):
                if "db_session" in cleaned_chunk["agent_output"]:
                    del cleaned_chunk["agent_output"]["db_session"]
                if "current_user" in cleaned_chunk["agent_output"]:
                    del cleaned_chunk["agent_output"]["current_user"]
            return cleaned_chunk

        # Initial thoughts and context setup
        thoughts.append(f"Orchestrator Agent received task: '{task}'")
        thoughts.append("Analyzing task and delegating to specialized agents.")
        thoughts.append("Delegating to Reasoning/Planning Agent for initial planning.")
        yield clean_chunk({"thoughts": thoughts, "status": "processing"})

        reasoning_agent = self.agents.get("Reasoning/Planning Agent")
        if not reasoning_agent:
            yield clean_chunk({"error": "Reasoning/Planning Agent not registered.", "status": "error"})
            return

        # Pass db_session and current_user to the reasoning agent
        reasoning_context = {
            **context,
            "db_session": db_session,
            "current_user": current_user,
            "search_ai_active": context.get("search_ai_active", True),
            "search_rag_active": context.get("search_rag_active", False),
            "search_online_active": context.get("search_online_active", False),
            "files": context.get("files"), # Pass files to reasoning context
        }

        # Handle file processing if files are present and agent is registered
        if context.get("files"):
            file_processing_agent = self.agents.get("Conversation Attachment File Processing Agent")
            if file_processing_agent:
                thoughts.append("Files detected. Delegating to File Processing Agent.")
                yield clean_chunk({"thoughts": thoughts, "status": "processing"})
                file_processing_context = {
                    "files": context["files"],
                    "db_session": db_session,
                    "current_user": current_user,
                    "user_prompt": task,
                }
                file_processing_result = await file_processing_agent.process("process_attached_files", file_processing_context)
                thoughts.extend(file_processing_result.get("thoughts", []))
                reasoning_context["processed_files_info"] = file_processing_result.get("output", {}).get("processed_files", [])
                yield clean_chunk({"thoughts": thoughts, "status": "processing", "agent_output": file_processing_result.get("output", {}), "agent_name": "File Processing Agent"})
            else:
                thoughts.append("File Processing Agent not registered, skipping file processing.")
                yield clean_chunk({"thoughts": thoughts, "status": "processing"})

        # Handle RAG search if active and agent is registered
        if context.get("search_rag_active"):
            rag_agent = self.agents.get("Information Retrieval Agent")
            if rag_agent:
                thoughts.append("RAG search active. Delegating to Information Retrieval Agent.")
                yield clean_chunk({"thoughts": thoughts, "status": "processing"})
                rag_context = {
                    "query": task, # Use the user's task as the query
                    "db_session": db_session,
                    "current_user": current_user,
                    "retrievable_rag_ids": context.get("retrievable_rag_ids", []),
                    "display_thoughts": context.get("display_thoughts", False),
                }
                rag_result = await rag_agent.process("query_rag", rag_context)
                thoughts.extend(rag_result.get("thoughts", []))
                reasoning_context["rag_results"] = rag_result.get("output", {})
                yield clean_chunk({"thoughts": thoughts, "status": "processing", "agent_output": rag_result.get("output", {}), "agent_name": "Information Retrieval Agent"})
            else:
                thoughts.append("RAG Agent not registered, skipping RAG search.")
                yield clean_chunk({"thoughts": thoughts, "status": "processing"})

        # Handle Online search if active and agent is registered
        if context.get("search_online_active"):
            search_agent = self.agents.get("Search Agent")
            if search_agent:
                thoughts.append("Online search active. Delegating to Search Agent.")
                yield clean_chunk({"thoughts": thoughts, "status": "processing"})
                search_context = {
                    "query": task, # Use the user's task as the query
                    "current_user": current_user,
                }
                search_result = await search_agent.process("perform_search", search_context)
                thoughts.extend(search_result.get("thoughts", []))
                reasoning_context["online_search_results"] = search_result.get("output", {}).get("results", [])
                yield clean_chunk({"thoughts": thoughts, "status": "processing", "agent_output": search_result.get("output", {}), "agent_name": "Search Agent"})
            else:
                thoughts.append("Search Agent not registered, skipping online search.")
                yield clean_chunk({"thoughts": thoughts, "status": "processing"})
        
        # The Reasoning/Planning Agent will return a plan and initial raw output
        reasoning_result = await reasoning_agent.process(task, reasoning_context)
        yield clean_chunk({"thoughts": thoughts, "status": "processing", "agent_output": reasoning_result, "agent_name": "Reasoning/Planning Agent"})

        # Pass through Validation Agent if available
        if "Validation Agent" in self.agents:
            thoughts.append("Passing reasoning output to Validation Agent.")
            yield clean_chunk({"thoughts": thoughts, "status": "processing"})
            validation_context = {
                "output_to_validate": reasoning_result,
                "user_prompt": task,
                "context": reasoning_context, # Pass full context including db_session, current_user
                "agent_name": "Reasoning/Planning Agent"
            }
            validation_result = await self.agents["Validation Agent"].process("validate_output", validation_context)
            yield clean_chunk({"thoughts": thoughts, "status": "processing", "agent_output": validation_result, "agent_name": "Validation Agent"})
            
            if validation_result.get("status") == "success":
                final_output = validation_result.get("validated_output", {})
                thoughts.append("Output validated successfully.")
            else:
                final_output = {"error": validation_result.get("error", "Validation failed.")}
                thoughts.append(f"Validation failed: {final_output['error']}")
        else:
            final_output = reasoning_result # If no validation agent, use raw output
            thoughts.append("No Validation Agent registered, skipping validation.")
        
        yield clean_chunk({"thoughts": thoughts, "status": "processing", "intermediate_output": final_output})

        # After processing, pass to Formatting/Output Agent for final formatting and streaming
        if "Formatting/Output Agent" in self.agents:
            thoughts.append("Passing final output to Formatting/Output Agent for streaming.")
            yield clean_chunk({"thoughts": thoughts, "status": "processing"})
            
            # The Formatting/Output Agent will handle conditional display of thoughts and stream its output
            formatted_response = await self.agents["Formatting/Output Agent"].process("format_response", {
                "raw_output": final_output,
                "orchestrator_thoughts": thoughts,
                "display_thoughts": context.get("display_thoughts", False), # Controlled by Orchestrator/user
                # Pass other agents' thoughts for comprehensive display if needed
                "reasoning_thoughts": reasoning_result.get("thoughts", []), # Assuming reasoning_result might contain its own thoughts
                "validation_thoughts": validation_result.get("thoughts", []) if "Validation Agent" in self.agents else [],
                "rag_thoughts": rag_result.get("thoughts", []) if context.get("search_rag_active") and self.agents.get("Information Retrieval Agent") else [],
                "file_processing_thoughts": file_processing_result.get("thoughts", []) if context.get("files") and self.agents.get("Conversation Attachment File Processing Agent") else [],
                "search_thoughts": search_result.get("thoughts", []) if context.get("search_online_active") and self.agents.get("Search Agent") else [],
                "code_generation_thoughts": [],
                "code_execution_thoughts": [],
                "workflow_control_thoughts": [],
            })
            yield clean_chunk({"output": formatted_response.get("output", ""), "thoughts": formatted_response.get("thoughts", []), "status": "completed"})
        else:
            final_output["error"] = final_output.get("error", "") + " Formatting/Output Agent not registered. Raw output returned."
            yield clean_chunk({"output": final_output, "thoughts": thoughts, "status": "completed"})
