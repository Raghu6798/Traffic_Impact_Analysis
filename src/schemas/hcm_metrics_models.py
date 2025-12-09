from pydantic import BaseModel
from typing import Dict, Any, List

class HCMReportMetrics(BaseModel):
    Average_Delay_sec: float
    Level_of_Service: str 
    Total_Vehicles_Processed: int
    Volume_to_Capacity_Ratio: float

class QueueReportDetails(BaseModel):
    edge_queues_95th_percentile: Dict[str, float] 

class SimulationReport(BaseModel):
    metrics: HCMReportMetrics
    details: Dict[str, Any] 
    

class TiaResponse(BaseModel):
    status: str 
    job_id: str
    report_data: SimulationReport

class SimulationRequest(BaseModel):
    # This is a placeholder. In the FastAPI route, we use Form() for file/text.
    # This model is more useful for the Job Payload sent to SQS.
    s3_key: str
    map_bbox: Dict[str, float]
    target_streets: str

class JobStatusResponse(BaseModel):
    status: str # PENDING, PROCESSING, COMPLETED, FAILED
    job_id: str
    report_data: Dict[str, Any] | None = None # The final report if COMPLETED