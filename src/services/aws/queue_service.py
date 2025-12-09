import boto3
import json
import os
from botocore.exceptions import ClientError
from typing import Dict, Any

from src.utils.logger import logger 

try:
    sqs_client = boto3.client('sqs')
except Exception as e:
    logger.error(f"Failed to initialize SQS client. Check AWS credentials. Error: {e}")
    sqs_client = None


class SQSQueueService:
    """
    Service for sending and receiving messages from AWS SQS.
    """
    def __init__(self, queue_url: str = os.getenv("SQS_QUEUE_URL")):
        if not sqs_client:
            raise RuntimeError("SQS Client not initialized. Check AWS config.")
        self.client = sqs_client
        self.queue_url = queue_url
        if not self.queue_url:
            logger.warning("SQS_QUEUE_URL environment variable is not set.")

    def push_job(self, job_id: str, payload: Dict[str, Any]) -> str:
        """
        Sends a job message to the SQS queue.
        
        Args:
            job_id: Unique identifier for the job.
            payload: Dictionary containing job details (e.g., file paths, BBox).
            
        Returns:
            The MessageId of the sent message.
        """
        if not self.queue_url:
            raise ValueError("SQS Queue URL is not configured.")

        job_message = {"job_id": job_id, **payload}
        
        try:
            response = self.client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(job_message),
                MessageAttributes={
                    'JobId': {
                        'DataType': 'String',
                        'StringValue': job_id
                    }
                }
            )
            message_id = response.get('MessageId')
            logger.info(f"Job {job_id} pushed to SQS. MessageId: {message_id}")
            return message_id
        except ClientError as e:
            logger.error(f"Failed to send message to SQS for job {job_id}. Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to enqueue job.") from e

    def receive_job(self, visibility_timeout: int = 600) -> list:
        """
        Receives messages from SQS. This is typically used by the background worker.
        
        Args:
            visibility_timeout: Time (in seconds) the message will be invisible to other consumers (default 10 mins).
            
        Returns:
            A list of messages (SQS format).
        """
        try:
            response = self.client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=visibility_timeout
            )
            return response.get('Messages', [])
        except ClientError as e:
            logger.error(f"Error receiving SQS message: {e}")
            return []

    def delete_job(self, receipt_handle: str) -> None:
        """Deletes a message from the queue after processing."""
        try:
            self.client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info(f"SQS message deleted.")
        except ClientError as e:
            logger.error(f"Error deleting SQS message: {e}")



sqs_service = SQSQueueService()

