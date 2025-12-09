# src/api/v1/routes.py

import os
from uuid import uuid4
from fastapi import APIRouter, UploadFile, Form, HTTPException
from typing import Dict, Any
from src.services.aws.queue_service import sqs_service
from src.services.aws.storage_service import s3_service
from src.utils.logger import logger

router = APIRouter()

from src.agent.executor import agent
@router.post("/chat-test")
async def chat_agent_test(query_msg: str) -> str:
    response_object = agent.invoke({"messages": [{"role": "user", "content": query_msg}]})
    return response_object['messages'][-1].content



@router.post("/simulate", summary="Submit a new TIA simulation job to SQS")
async def submit_simulation_job(
    file: UploadFile = File(..., description="The TMC volume count CSV/Excel file."),
    north: float = Form(..., description="North Bounding Box coordinate."),
    south: float = Form(..., description="South Bounding Box coordinate."),
    east: float = Form(..., description="East Bounding Box coordinate."),
    west: float = Form(..., description="West Bounding Box coordinate."),
    target_streets: str = Form(..., description="Comma-separated street names for mapping.")
) -> Dict[str, str]:
    """
    Receives simulation inputs, uploads the file to S3, and pushes a job to SQS.
    Returns immediately with a Job ID and Queue Status.
    """
    job_id = str(uuid4())
    temp_dir = f"/tmp/uploads/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    local_file_path = os.path.join(temp_dir, file.filename)

    try:
        file_content = await file.read()
        with open(local_file_path, "wb") as buffer:
            buffer.write(file_content)
    except Exception as e:
        logger.error(f"File write failed: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded file temporarily.")

    s3_key = f"input/{job_id}/{file.filename}"

    try:
        s3_service.upload_file(local_file_path, s3_key)
    except Exception as e:
        logger.error(f"S3 Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to S3.")
    finally:
        # 3. Clean up the temporary local file immediately
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        os.rmdir(temp_dir)

    payload: Dict[str, Any] = {
        "s3_key": s3_key,
        "map_bbox": {"n": north, "s": south, "e": east, "w": west},
        "target_streets": target_streets
    }

    try:
        sqs_message_id = sqs_service.push_job(job_id, payload)
        return {
            "status": "Job Queued",
            "job_id": job_id,
            "sqs_message_id": sqs_message_id
        }
    except Exception as e:
        # The SQS push_job function already raises HTTPException on ClientError
        # This catches other errors (e.g., misconfiguration)
        logger.error(f"Critical SQS enqueue failure: {e}")
        raise HTTPException(status_code=503, detail="Queue service unavailable.")