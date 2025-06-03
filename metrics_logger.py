import ray
from google.cloud import storage
import json
import os
import logging
from utils import parse_gcs_uri, InvalidGCSPathError # Added

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(self, gcs_log_dir: str):
        """
        Initializes the MetricsLogger Actor.

        Args:
            gcs_log_dir (str): The GCS directory path (e.g., 'gs://your-results-bucket/logs/')
                               where task metrics will be stored.
        """
        if not gcs_log_dir.startswith("gs://"): # Initial check retained for immediate feedback
            logger.error(f"gcs_log_dir must be a GCS path starting with 'gs://'. Got: {gcs_log_dir}")
            raise ValueError("gcs_log_dir must be a GCS path starting with 'gs://'")

        # self.gcs_log_dir = gcs_log_dir.rstrip('/') + '/' # Ensure trailing slash for consistency

        try:
            # Parse the original gcs_log_dir first
            raw_bucket_name, raw_blob_prefix = parse_gcs_uri(gcs_log_dir)

            self.bucket_name = raw_bucket_name
            # Ensure blob_prefix has a trailing slash if it's not empty for consistency
            if raw_blob_prefix:
                self.blob_prefix = raw_blob_prefix.rstrip('/') + '/'
            else:
                self.blob_prefix = "" # Handles cases like "gs://bucket-name" or "gs://bucket-name/"

            # Reconstruct gcs_log_dir for internal use, ensuring it ends with a slash
            self.gcs_log_dir = f"gs://{self.bucket_name}/{self.blob_prefix}"
            if not self.gcs_log_dir.endswith('/'): # Should already be handled but as a safeguard
                 self.gcs_log_dir += '/'

        except InvalidGCSPathError as e:
            logger.error(f"Invalid GCS log directory URI in MetricsLogger: {gcs_log_dir} - {e}")
            raise # Re-raise to prevent actor from starting with bad config

        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        logger.info(f"MetricsLogger initialized. Logging to: {self.gcs_log_dir} (Bucket: {self.bucket_name}, Prefix: {self.blob_prefix})")

    def log_task_metrics(self, metrics: dict):
        """
        Receives and logs detailed task results to a JSON file in GCS.

        Args:
            metrics (dict): A dictionary containing all relevant task metrics
                            (e.g., task_id, duration, cost, utilization, etc.).
        """
        task_id = metrics.get('task_id', 'unknown_task')
        file_name = f"{task_id}.json"
        blob_path = os.path.join(self.blob_prefix, file_name)

        try:
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(json.dumps(metrics, indent=2), content_type="application/json")
            logger.info(f"Logged metrics for task {task_id} to {self.gcs_log_dir}{file_name}")
        except Exception as e:
            logger.exception(f"Error logging metrics for task {task_id} to GCS:")

    def get_all_logged_task_ids(self):
        """
        (Optional) Retrieves a list of all task IDs that have been logged.
        Useful for debugging/verification.
        """
        logged_ids = []
        for blob in self.bucket.list_blobs(prefix=self.blob_prefix):
            if blob.name.endswith('.json'):
                task_id = os.path.basename(blob.name).replace('.json', '')
                logged_ids.append(task_id)
        return logged_ids