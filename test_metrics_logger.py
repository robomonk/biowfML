import unittest
from unittest.mock import patch, MagicMock
import json
import os

# Assuming metrics_logger.py and utils.py are in the same directory or accessible via PYTHONPATH
from metrics_logger import MetricsLogger
from utils import InvalidGCSPathError

class TestMetricsLogger(unittest.TestCase):

    @patch('metrics_logger.storage.Client')
    def test_constructor_valid_gcs_path(self, mock_storage_client):
        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        logger_instance = MetricsLogger(gcs_log_dir="gs://my-bucket/logs/")
        self.assertEqual(logger_instance.gcs_log_dir, "gs://my-bucket/logs/")
        self.assertEqual(logger_instance.bucket_name, "my-bucket")
        self.assertEqual(logger_instance.blob_prefix, "logs/")
        mock_storage_client.return_value.bucket.assert_called_once_with("my-bucket")

    @patch('metrics_logger.storage.Client')
    def test_constructor_valid_gcs_path_bucket_only(self, mock_storage_client):
        mock_bucket = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        logger_instance = MetricsLogger(gcs_log_dir="gs://my-bucket/")
        self.assertEqual(logger_instance.gcs_log_dir, "gs://my-bucket/")
        self.assertEqual(logger_instance.bucket_name, "my-bucket")
        self.assertEqual(logger_instance.blob_prefix, "")
        mock_storage_client.return_value.bucket.assert_called_once_with("my-bucket")

    def test_constructor_invalid_gcs_path(self):
        with self.assertRaisesRegex(ValueError, "gcs_log_dir must be a GCS path starting with 'gs://'"):
            MetricsLogger(gcs_log_dir="s3://my-bucket/logs/")

        with self.assertRaisesRegex(InvalidGCSPathError, "Bucket name is empty"):
            MetricsLogger(gcs_log_dir="gs:///")

        with self.assertRaisesRegex(InvalidGCSPathError, "Path after 'gs://' is empty"):
            MetricsLogger(gcs_log_dir="gs://")

    @patch('metrics_logger.storage.Client')
    def test_log_task_metrics(self, mock_storage_client):
        mock_blob_instance = MagicMock()
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob_instance
        mock_storage_client.return_value.bucket.return_value = mock_bucket_instance

        logger_instance = MetricsLogger(gcs_log_dir="gs://test-bucket/test-logs/")

        metrics_data = {"task_id": "task123", "value": 42, "status": "SUCCESS"}
        logger_instance.log_task_metrics(metrics_data)

        expected_blob_path = "test-logs/task123.json"
        mock_bucket_instance.blob.assert_called_once_with(expected_blob_path)

        args, kwargs = mock_blob_instance.upload_from_string.call_args
        self.assertEqual(json.loads(args[0]), metrics_data)
        self.assertEqual(kwargs['content_type'], "application/json")

    @patch('metrics_logger.storage.Client')
    def test_log_task_metrics_no_task_id(self, mock_storage_client):
        mock_blob_instance = MagicMock()
        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob_instance
        mock_storage_client.return_value.bucket.return_value = mock_bucket_instance

        logger_instance = MetricsLogger(gcs_log_dir="gs://test-bucket/test-logs/")

        metrics_data = {"value": 42, "status": "SUCCESS"}
        logger_instance.log_task_metrics(metrics_data)

        expected_blob_path = "test-logs/unknown_task.json"
        mock_bucket_instance.blob.assert_called_once_with(expected_blob_path)
        args, kwargs = mock_blob_instance.upload_from_string.call_args
        self.assertEqual(json.loads(args[0]), metrics_data)
        self.assertEqual(kwargs['content_type'], "application/json")


    @patch('metrics_logger.storage.Client')
    def test_get_all_logged_task_ids(self, mock_storage_client):
        mock_bucket_instance = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket_instance

        blob1 = MagicMock()
        blob1.name = "test-logs/task001.json"
        blob2 = MagicMock()
        blob2.name = "test-logs/subfolder/task002.json"
        blob3 = MagicMock()
        blob3.name = "test-logs/task003.txt"
        blob4 = MagicMock()
        blob4.name = "test-logs/task004.json"

        mock_bucket_instance.list_blobs.return_value = [blob1, blob2, blob3, blob4]

        logger_instance = MetricsLogger(gcs_log_dir="gs://test-bucket/test-logs/")
        logged_ids = logger_instance.get_all_logged_task_ids()

        mock_bucket_instance.list_blobs.assert_called_once_with(prefix="test-logs/")

        self.assertCountEqual(logged_ids, ["task001", "task002", "task004"])

if __name__ == '__main__':
    unittest.main()
