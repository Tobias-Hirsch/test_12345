import json
from typing import Any, Dict, Optional
from .base_agent import BaseAgent
from app.rag_knowledge.generic_knowledge import query_rag_system
from app.models.database import RagData
from sqlalchemy.orm import Session

class InformationRetrievalAgent(BaseAgent):
    """
    The Information Retrieval Agent is responsible for interacting with the RAG system.
    It receives queries, retrieves relevant information from the knowledge base,
    and processes it for integration into the overall response.
    """

    def __init__(self):
        super().__init__(name="Information Retrieval Agent",
                         description="Retrieves and processes information from RAG knowledge base.")

    async def process(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query by retrieving information from the RAG system.
        This agent now acts as a wrapper around the query_rag_system function,
        which encapsulates all the logic for permission checks, multi-collection search,
        and response synthesis.

        Args:
            task (str): The specific query or instruction for RAG.
            context (Dict[str, Any]): Contextual information, including user prompt,
                                     db_session, and retrievable_rag_ids.

        Returns:
            Dict[str, Any]: A dictionary containing the final answer and source documents.
        """
        thoughts = []
        output = {}
        query = context.get("query", task)
        db_session: Session = context.get("db_session")
        retrievable_rag_ids = context.get("retrievable_rag_ids", [])
        show_think_process = context.get("display_thoughts", False) # Get the flag from context

        if not db_session:
            raise ValueError("db_session not found in context for InformationRetrievalAgent.")

        if not retrievable_rag_ids:
            thoughts.append("Permission check: User has no access to any RAG items for retrieval.")
            output["answer"] = "You do not have permission to access any knowledge base."
            output["source_documents"] = []
            return {"output": output, "thoughts": thoughts}

        thoughts.append(f"Information Retrieval Agent received task: '{task}' with query: '{query}'")
        thoughts.append(f"User has access to RAG IDs: {retrievable_rag_ids}")
        
        try:
            # Fetch RAG item details (id and name) from the database
            permitted_rag_items = db_session.query(RagData.id, RagData.name).filter(RagData.id.in_(retrievable_rag_ids)).all()
            permitted_rag_items_dict = [{"id": item.id, "name": item.name} for item in permitted_rag_items]
            
            thoughts.append(f"Found {len(permitted_rag_items_dict)} permitted RAG items to query.")
            # Call the centralized RAG query system
            # This function now handles everything: searching, ranking, and synthesizing the answer.
            answer_parts = []
            source_documents = []
            rag_result_stream = query_rag_system(
                query_text=query,
                db_session=db_session,
                permitted_rag_items=permitted_rag_items_dict,
                show_think_process=show_think_process # Pass the flag here
            )
            async for chunk in rag_result_stream:
                if isinstance(chunk, str):
                    answer_parts.append(chunk)
                elif isinstance(chunk, dict):
                    if chunk.get("answer"):
                        answer_parts.append(chunk["answer"])
                    if "source_documents" in chunk:
                        source_documents.extend(chunk["source_documents"])

            answer = "".join(answer_parts).strip()
            if answer or source_documents:
                output["answer"] = answer or "No direct answer was generated from the knowledge base."
                output["source_documents"] = source_documents
                thoughts.append("Successfully retrieved and synthesized response from RAG system.")
            else:
                thoughts.append("RAG system returned no results.")
                output["answer"] = "No relevant information found in the knowledge base."
                output["source_documents"] = []

        except Exception as e:
            thoughts.append(f"Error during RAG query system call: {e}")
            output["error"] = f"An error occurred during RAG retrieval: {e}"
            output["answer"] = f"An error occurred during RAG retrieval: {e}"
            output["source_documents"] = []

        thoughts.append("Information Retrieval complete.")
        return {"output": output, "thoughts": thoughts}
