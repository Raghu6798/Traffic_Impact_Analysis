from fastapi import APIRouter, Depends, HTTPException
from src.agent.executor import agent

router = APIRouter()


@router.post("/api/v1/agent")
async def chat_agent(query_msg:str)->str:
    response_object = agent.invoke({"messages":[{"role":"user","content":query_msg}]})
    return response_object['messages'][-1].content