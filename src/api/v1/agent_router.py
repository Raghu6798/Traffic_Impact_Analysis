import shutil
import os

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, List
from src.agent.executor import agent
from src.utils.logger import logger


router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    context: Optional[Dict[str, Any]] = None 


class ChatResponse(BaseModel):
    reply: str

@router.post("/api/v1/agent", response_model=ChatResponse)
async def chat_agent(request: ChatRequest):
    try:

        input_message = {"role": "user", "content": request.message}
        
        if request.context:
            context_str = f"\n\n[System Context: {request.context}]"
            input_message["content"] += context_str


        response_object = agent.invoke(
            {"messages": [input_message]},
            {"configurable": {"thread_id": request.thread_id}} 
        )
        
    
        raw_content = response_object['messages'][-1].content
        
        reply_text = ""
        
        if isinstance(raw_content, str):
            reply_text = raw_content
        elif isinstance(raw_content, list):
            reply_text = "".join([
                block.get("text", "") 
                for block in raw_content 
                if isinstance(block, dict) and block.get("type") == "text"
            ])
        else:
            reply_text = str(raw_content)

        return {"reply": reply_text}
        
    except Exception as e:
        logger.error(f"Agent Error: {e}")
        return {"reply": f"System Error: {str(e)}"}


@router.post("/api/v1/upload_traffic_data")
async def upload_traffic_data(file: UploadFile = File(...)):
    try:
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_location = f"{upload_dir}/{file.filename}"
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": file.filename, "status": "success", "filepath": file_location}
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
