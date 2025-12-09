from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agent.executor import agent
from typing import Optional, Dict, Any

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    context: Optional[Dict[str, Any]] = None 

@router.post("/api/v1/agent")
async def chat_agent(request: ChatRequest) -> str:
    try:
    
        input_message = {"role": "user", "content": request.message}
        
        if request.context:
            context_str = f"\n\n[System Context: {request.context}]"
            input_message["content"] += context_str

   
        response_object = agent.invoke(
            {"messages": [input_message]},
            {"configurable": {"thread_id": request.thread_id}} 
        )
        
        return response_object['messages'][-1].content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))