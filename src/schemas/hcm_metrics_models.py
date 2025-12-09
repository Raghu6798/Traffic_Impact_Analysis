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