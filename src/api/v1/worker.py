# src/worker.py

import json
import os
import time
from src.services.aws.queue_service import sqs_service
from src.services.aws.storage_service import s3_service
from src.utils.logger import logger
from src.agent.executor import agent # Import the agent instance
from src.config.settings import get_settings
from src.models.schemas import SimulationReport # Assuming you map the final output to this

# NOTE: You need to implement DB updates (Prisma) in a real worker
# from src.services.database_service import update_job_status, save_report

settings = get_settings()

def execute_tia_agent_workflow(job_payload: dict, work_dir: str):
    """
    Executes the LangChain Agent workflow based on SQS job payload.
    """
    job_id = job_payload['job_id']
    logger.info(f"--- Starting TIA Agent Workflow for Job {job_id} ---")
    
    # Update DB Status: PROCESSING (if using DB)

    # 1. Download Input File
    s3_key = job_payload['s3_key']
    input_filename = os.path.basename(s3_key)
    local_input_file = os.path.join(work_dir, input_filename)
    s3_service.download_file(s3_key, local_input_file)
    
    # 2. Construct Agent Prompt using inputs
    bbox = job_payload['map_bbox']
    streets = job_payload['target_streets']
    
    agent_instruction = f"""
    Execute the full TIA workflow for Job ID {job_id}. 
    1. Download map data using BBox: N={bbox['n']}, S={bbox['s']}, E={bbox['e']}, W={bbox['w']}.
    2. Use the traffic volume data found in the file located at: {local_input_file}.
    3. Map the topology using the street names: {streets}.
    4. Run the full simulation steps (Netconvert, JTRRouter, Sumo) and finally call compute_hcm_metrics and parse_queue_xml.
    5. Provide the final, unified report as your last step.
    """
    
    # 3. EXECUTE THE LANGCHAIN AGENT
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": agent_instruction}]})
        final_output_str = result.get('output', 'Agent finished but returned no final output.')
        
        # The Agent should return a JSON string from compute_hcm_metrics
        final_report_json = json.loads(final_output_str)
        
        # 4. Save Report to S3 and DB
        # s3_service.upload_file(..., f"results/{job_id}/final_report.json")
        # save_report(job_id, "COMPLETED", final_report_json)
        
        logger.success(f"Workflow for Job {job_id} finished successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Agent Execution FAILED for Job {job_id}: {e}")
        # Update DB Status: FAILED
        return False

# --- Main Worker Loop ---
def worker_main():
    """
    The main loop that listens to SQS and processes jobs.
    This is the CMD entry point for your Docker Container.
    """
    logger.info("TIA Worker starting. Listening to SQS...")
    WORK_DIR = os.getenv("WORK_DIR", "/tmp/sim_work")
    
    while True:
        messages = sqs_service.receive_job(visibility_timeout=900) # Increased to 15 min
        
        if messages:
            for message in messages:
                receipt_handle = message['ReceiptHandle']
                try:
                    job_payload = json.loads(message['Body'])
                    job_id = job_payload.get('job_id', str(uuid4()))
                    
                    logger.info(f"--- Processing Job: {job_id} ---")
                    os.makedirs(WORK_DIR, exist_ok=True) # Ensure work dir exists
                    
                    success = execute_tia_agent_workflow(job_payload, WORK_DIR)
                    
                    if success:
                        sqs_service.delete_job(receipt_handle)
                        logger.info(f"--- Job {job_id} COMPLETED ---")
                    else:
                        logger.warning(f"Job {job_id} failed execution. Allowing SQS to retry.")
                        
                except Exception as e:
                    logger.error(f"FATAL SYSTEM ERROR for job: {e}")
                    # DO NOT DELETE MESSAGE: SQS handles visibility timeout/retry
                    
        else:
            time.sleep(5) 

if __name__ == "__main__":
    worker_main()