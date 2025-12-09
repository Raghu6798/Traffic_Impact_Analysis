from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, List
from src.agent.executor import agent
from src.utils.logger import logger

router = APIRouter()

# 1. Define Request Model
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    context: Optional[Dict[str, Any]] = None 

# 2. Define Response Model
class ChatResponse(BaseModel):
    reply: str

@router.post("/api/v1/agent", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    try:
        # Prepare Input
        input_message = {"role": "user", "content": request.message}
        
        if request.context:
            context_str = f"\n\n[System Context: {request.context}]"
            input_message["content"] += context_str

        # Invoke Agent
        response_object = agent.invoke(
            {"messages": [input_message]},
            {"configurable": {"thread_id": request.thread_id}} 
        )
        
        # --- CRITICAL FIX START ---
        # Handle Gemini returning a list of objects instead of a plain string
        raw_content = response_object['messages'][-1].content
        
        reply_text = ""
        
        if isinstance(raw_content, str):
            reply_text = raw_content
        elif isinstance(raw_content, list):
            # Join all text blocks into one string
            reply_text = "".join([
                block.get("text", "") 
                for block in raw_content 
                if isinstance(block, dict) and block.get("type") == "text"
            ])
        else:
            reply_text = str(raw_content)
        # --- CRITICAL FIX END ---

        return {"reply": reply_text}
        
    except Exception as e:
        logger.error(f"Agent Error: {e}")
        # Return the error message to the chat window so the user knows what happened
        return {"reply": f"System Error: {str(e)}"}