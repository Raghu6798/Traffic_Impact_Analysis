import os
import boto3
from botocore.exceptions import ClientError

try:
    s3_client = boto3.client('s3')
except Exception as e:
    logger.error(f"Failed to initialize S3 client. Check AWS credentials. Error: {e}")
    s3_client = None

class S3StorageService:
    """
    Service for handling file operations with AWS S3.
    """
    def __init__(self, bucket_name: str = os.getenv("S3_BUCKET_NAME")):
        if not s3_client:
            raise RuntimeError("S3 Client not initialized. Check AWS config.")
        self.client = s3_client
        self.bucket_name = bucket_name
        if not self.bucket_name:
            logger.warning("S3_BUCKET_NAME environment variable is not set.")

    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Uploads a local file to S3.
        
        Args:
            file_path: The local path to the file to upload (e.g., '/tmp/data.csv').
            object_name: The S3 key (path) to store the file as (e.g., 'jobs/123/data.csv').
            
        Returns:
            The public S3 URL (or key) of the uploaded file.
        """
        if not self.bucket_name:
            raise ValueError("S3 Bucket Name is not configured.")
        
        try:
            self.client.upload_file(file_path, self.bucket_name, object_name)
            logger.info(f"File {file_path} uploaded to S3://{self.bucket_name}/{object_name}")
            # Returns a simplified URL/key for use
            return f"s3://{self.bucket_name}/{object_name}"
        except ClientError as e:
            logger.error(f"Failed to upload file {file_path} to S3. Error: {e}")
            raise e

    def download_file(self, object_name: str, download_path: str) -> bool:
        """
        Downloads a file from S3 to a local path. Used by the background worker.
        
        Args:
            object_name: The S3 key of the file to download.
            download_path: The local path where the file should be saved.
            
        Returns:
            True on success, False otherwise.
        """
        if not self.bucket_name:
            raise ValueError("S3 Bucket Name is not configured.")

        try:
            # Ensure the local directory exists
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            
            self.client.download_file(self.bucket_name, object_name, download_path)
            logger.info(f"File {object_name} downloaded from S3 to {download_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download file {object_name} from S3. Error: {e}")
            return False

s3_service = S3StorageService()