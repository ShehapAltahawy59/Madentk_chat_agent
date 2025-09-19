from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from agent.client import az_model_client
from agent.agent import smart_agent, AGENT_TOOLS
from agent.prompt import system_message
from agent import tools as tools_module

# Simple in-memory session context store
# Maps user_id -> {"where": str}
SESSION_CTX = {}

# Agent cache to avoid creating new agents for each request
AGENT_CACHE = {}
MAX_CACHED_AGENTS = 100  # Limit to prevent memory bloat

def cleanup_agent_cache():
    """Clean up agent cache if it gets too large"""
    if len(AGENT_CACHE) > MAX_CACHED_AGENTS:
        # Remove oldest agents (simple FIFO for now)
        oldest_users = list(AGENT_CACHE.keys())[:len(AGENT_CACHE) - MAX_CACHED_AGENTS + 10]
        for user_id in oldest_users:
            del AGENT_CACHE[user_id]
        print(f"🧹 Cleaned up agent cache, removed {len(oldest_users)} agents")

class ChatRequest(BaseModel):
    user_query: str
    history: List[List[Optional[str]]] = []
    user_id: Optional[str] = None
    where: Optional[str] = None

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    user_query = request.user_query
    message_with_context = user_query

    # Resolve session context: prefer provided values, else reuse stored, else defaults
    resolved_user_id = request.user_id
    resolved_where = request.where
    if resolved_user_id:
        # Initialize session bucket if missing
        if resolved_user_id not in SESSION_CTX:
            SESSION_CTX[resolved_user_id] = {}
        # If where provided, store/update; else reuse previous if any
        if resolved_where:
            SESSION_CTX[resolved_user_id]["where"] = resolved_where
        else:
            resolved_where = SESSION_CTX[resolved_user_id].get("where", None)

    # Set defaults if still missing
    if not resolved_where:
        resolved_where = "quweisna"

    # Store active user id and where (available to tools and agent via tool)
    tools_module.set_active_user_id(resolved_user_id)
    tools_module.set_active_where(resolved_where)

    # Get or create user-specific agent with memory
    if resolved_user_id:
        try:
            # Check if agent is already cached
            if resolved_user_id in AGENT_CACHE:
                user_agent = AGENT_CACHE[resolved_user_id]
                print(f"🔍 Using cached agent for user {resolved_user_id}")
            else:
                # Load user memory and create new agent
                await tools_module.user_memory_manager.load_user_memories(resolved_user_id)
                user_memory = tools_module.user_memory_manager.get_user_memory(resolved_user_id)
                
                # Create user-specific agent with memory
                user_agent = AssistantAgent(
                    name=f"madentk_agent_{resolved_user_id}",
                    model_client=az_model_client,
                    system_message=system_message,
                    tools=AGENT_TOOLS,
                    memory=[user_memory],
                    reflect_on_tool_use=True,
                    max_tool_iterations=5
                )
                
                # Cache the agent
                AGENT_CACHE[resolved_user_id] = user_agent
                print(f"🔍 Created and cached user-specific agent for user {resolved_user_id}")
                
                # Clean up cache if needed
                cleanup_agent_cache()
        except Exception as e:
            print(f"⚠ Error loading user memory, using global agent: {e}")
            user_agent = smart_agent
    else:
        user_agent = smart_agent

    messages = []
    
    # Use provided history if available, otherwise let memory system handle context
    if request.history:
        for pair in request.history[-5:]:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                user_msg, assistant_msg = pair
                if user_msg and isinstance(user_msg, str):
                    messages.append(TextMessage(content=user_msg, source="user"))
                if assistant_msg and isinstance(assistant_msg, str):
                    messages.append(TextMessage(content=assistant_msg, source="assistant"))
        print(f"🔍 Using provided history with {len(messages)} messages")
    else:
        print(f"🔍 No history provided, memory system will provide context for user {resolved_user_id}")

    # Add user_id context to the main message instead of separate message
    if request.user_id:
        message_with_context = f"USER_ID={request.user_id}\n\n{message_with_context}"
        print(f"🔍 Added USER_ID context: {request.user_id}")

    messages.append(TextMessage(content=message_with_context, source="user"))

    # Call user-specific agent and ensure non-empty response
    try:
        response = await user_agent.on_messages(messages, cancellation_token=CancellationToken())
        content = getattr(response.chat_message, "content", "")
        if not content or not content.strip():
            print("Empty agent response; returning fallback.")
            return {"response": "آسف، حصلت مشكلة مؤقتة. جرب تاني لو سمحت."}
        # Note: Conversations are not automatically saved to Firebase
        # Only orders and preferences are saved when explicitly called by the agent
        
        print(f"Agent response: {content}")
        return {"response": content}
    except Exception as e:
        print(f"⚠ Error in agent response: {e}")
        return {"response": "آسف، فيه مشكلة في النظام. جرب تاني أو قولي اسم المطعم بالظبط!"} 
