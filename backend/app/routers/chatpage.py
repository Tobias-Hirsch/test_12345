from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel
from typing import List, Literal
from ..services.ollama_deepseek import get_deepseek_reply

router = APIRouter()

# Kommentar
user_messages = {}

class Message(BaseModel):
    id: str
    text: str
    sender: Literal['user', 'bot']

class MessageRequest(BaseModel):
    text: str

class MessageResponse(BaseModel):
    reply: str

@router.get("/api/messages/{user_id}", response_model=List[Message])
def get_messages(user_id: str = Path(...)):
    return user_messages.get(user_id, [])

@router.post("/api/messages/{user_id}", response_model=MessageResponse)
def post_message(user_id: str, req: MessageRequest):
    # Kommentar
    msg = Message(id=str(id(req)), text=req.text, sender='user')
    user_msgs = user_messages.setdefault(user_id, [])
    user_msgs.append(msg)
    # Kommentar
    try:
        reply = get_deepseek_reply(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    bot_msg = Message(id=str(id(reply)), text=reply, sender='bot')
    user_msgs.append(bot_msg)
    return MessageResponse(reply=reply)