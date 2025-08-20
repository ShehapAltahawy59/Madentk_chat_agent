from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from agent.agent import smart_agent

class ChatRequest(BaseModel):
    user_query: str
    history: List[List[Optional[str]]] = []
    user_id: Optional[str] = None

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.user_query
    history = request.history
    message_with_context = user_query

    messages = []
    for pair in history[-5:]:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            user_msg, assistant_msg = pair
            if user_msg and isinstance(user_msg, str):
                messages.append(TextMessage(content=user_msg, source="user"))
            if assistant_msg and isinstance(assistant_msg, str):
                messages.append(TextMessage(content=assistant_msg, source="assistant"))

    # Provide user_id as a context message if supplied
    if request.user_id:
        messages.append(TextMessage(content=f"USER_ID={request.user_id}", source="user"))

    messages.append(TextMessage(content=message_with_context, source="user"))

    # Call agent and ensure non-empty response
    try:
        response = await smart_agent.on_messages(messages, cancellation_token=CancellationToken())
        content = getattr(response.chat_message, "content", "")
        if not content or not content.strip():
            print("Empty agent response; returning fallback.")
            return {"response": "آسف، حصلت مشكلة مؤقتة. جرب تاني لو سمحت."}
        print(f"Agent response: {content}")
        return {"response": content}
    except Exception as e:
        print(f"⚠ Error in agent response: {e}")
        return {"response": "آسف، فيه مشكلة في النظام. جرب تاني أو قولي اسم المطعم بالظبط!"} 
