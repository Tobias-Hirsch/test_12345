# `routers/chat.py` - Kommentar

Hinweis`backend/app/routers/chat.py` Hinweis

## Kommentar
*   **SendenKommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **Kommentar**: Kommentar
*   **RAG Kommentar**: Kommentar
*   **LLM Kommentar**: Kommentar

## Kommentar
1.  **`add_conversation_message(message: schemas.AgentChatRequest, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_active_user))`**:
    *   Kommentar
    *   Kommentar`AGENTIC_RAG_ENABLE=False` Kommentar`rag_knowledge.generic_knowledge.query_rag_system` Kommentar
    *   Kommentar
    *   Kommentar`backend.app.llm.chain` MittelKommentar
    *   Kommentar
    *   Kommentar
    *   `@router.post("/add_conversation_message")`
2.  **`get_conversation_history(conversation_id: str, current_user: User = Depends(auth.get_current_active_user))`**:
    *   Kommentar`conversation_id` Kommentar
    *   Kommentar
    *   `@router.get("/history/{conversation_id}")`
3.  **`create_new_conversation(current_user: User = Depends(auth.get_current_active_user))`**:
    *   Kommentar
    *   Kommentar
    *   `@router.post("/new_conversation")`

## Kommentar
`/backend/app/routers/chat.py`