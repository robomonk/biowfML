import ray
from google.cloud import storage
import json
import os

@ray.remote
class MetricsLogger:
    def __init__(self, gcs_log_dir: str):
        """
        Initializes the MetricsLogger Actor.

        Args:
            gcs_log_dir (str): The GCS directory path (e.g., 'gs://your-results-bucket/logs/')
                               where task metrics will be stored.
        """
        if not gcs_log_dir.startswith("gs://"):
            raise ValueError("gcs_log_dir must be a GCS path starting with 'gs://'")
        self.gcs_log_dir = gcs_log_dir.rstrip('/') + '/'
        self.bucket_name = self.gcs_log_dir.split('//')[1].split('/')[0]
        self.blob_prefix = '/'.join(self.gcs_log_dir.split('/')[3:])
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)
        print(f"MetricsLogger initialized. Logging to: {self.gcs_log_dir}")

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
            print(f"Logged metrics for task {task_id} to {self.gcs_log_dir}{file_name}")
        except Exception as e:
            print(f"Error logging metrics for task {task_id} to GCS: {e}")

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