mport os
import json
from uuid import uuid4
from fastapi import APIRouter, UploadFile, Form, HTTPException, File
from typing import Dict, Any

# --- Imports based on your structure ---
from src.services.aws.queue_service import sqs_service
from src.services.aws.storage_service import s3_service
from src.utils.logger import logger
from src.utils.auth_utils import get_current_user 
from src.models.schemas import TiaResponse 

router = APIRouter()

@router.post("/simulate", status_code=202) 
async def submit_simulation_job(
    file: UploadFile = File(..., description="The TMC volume count CSV/Excel file."),
    north: float = Form(...),
    south: float = Form(...),
    east: float = Form(...),
    west: float = Form(...),
    target_streets: str = Form(...)
):
    """
    Receives simulation parameters, uploads the file to S3, and pushes a job to SQS.
    Returns immediately with a Job ID.
    """
    # 1. AUTH: Ensure user is logged in (You need to implement get_current_user)
    # user = await get_current_user() # Assuming a dependency for auth
    # if not user: raise HTTPException(status_code=401, detail="Unauthorized")
    
    job_id = str(uuid4())
    temp_dir = f"/tmp/uploads/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    local_file_path = os.path.join(temp_dir, file.filename)

    # 2. Save and Upload
    try:
        file_content = await file.read()
        with open(local_file_path, "wb") as buffer:
            buffer.write(file_content)
            
        s3_key = f"input/{job_id}/{file.filename}"
        s3_url = s3_service.upload_file(local_file_path, s3_key)
    except Exception as e:
        logger.error(f"Upload/File handling failed: {e}")
        raise HTTPException(status_code=500, detail="File processing failed.")
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    # 3. Create SQS Payload (This payload tells the worker what to do)
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "s3_key": s3_key,
        "map_bbox": {"n": north, "s": south, "e": east, "w": west},
        "target_streets": target_streets,
        # "user_id": user.id # Track who submitted the job
    }

    # 4. Push job to SQS
    try:
        sqs_service.push_job(job_id, payload)
        
        # Update DB status to PENDING here using Prisma (if you have the model)
        
        return {"status": "Job Queued", "job_id": job_id}
    except Exception as e:
        logger.error(f"Critical SQS enqueue failure: {e}")
        raise HTTPException(status_code=503, detail="Queue service unavailable.")