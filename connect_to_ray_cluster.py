import ray
from google.cloud import aiplatform_v1beta1 as aiplatform
from vertex_ray import get_ray_cluster
import time
import os
import argparse

# --- Configuration via Arguments ---
parser = argparse.ArgumentParser(description="Connect to an existing Vertex AI Ray cluster.")
parser.add_argument("--project-id", required=True, help="Your GCP Project ID")
parser.add_argument("--location", required=True, help="GCP region for the cluster (e.g., us-central1)")
parser.add_argument("--cluster-name", required=True, help="Name of the existing Ray cluster")
parser.add_argument("--code-gcs-path", required=True, help="GCS path to the code directory (e.g., gs://bucket/code/)")
args = parser.parse_args()

PROJECT_ID = args.project_id
LOCATION = args.location
CLUSTER_NAME = args.cluster_name
CODE_GCS_PATH = args.code_gcs_path

# --- 1. Get Info of the Existing Ray Cluster ---
print(f"Getting info for existing Ray cluster '{CLUSTER_NAME}' in {LOCATION} for project {PROJECT_ID}...")

try:
    cluster_info = get_ray_cluster(
        project_id=PROJECT_ID,
        location=LOCATION,
        cluster_name=CLUSTER_NAME
    )
    status = cluster_info.state.name
    print(f"Cluster status: {status}")

    if status != "RUNNING":
        print("Cluster is not RUNNING. Please check its status in Vertex AI Console.")
        # Optionally wait for it to become running
        # while status != "RUNNING":
        #     print(f"Cluster status: {status}... Waiting for RUNNING state.")
        #     time.sleep(30)
        #     cluster_info = get_ray_cluster(
        #         project_id=PROJECT_ID, location=LOCATION, cluster_name=CLUSTER_NAME
        #     )
        #     status = cluster_info.state.name
        # print("Cluster is now RUNNING.")


    # --- 2. Connect to the Existing Cluster ---
    print(f"Connecting to Ray cluster '{CLUSTER_NAME}'...")

    ray.init(
        address=f"vertex_ray://{CLUSTER_NAME}",
        runtime_env={
            "working_dir": CODE_GCS_PATH,
        }
    )
    print("Successfully connected to Ray cluster.")
    print(f"Ray Dashboard: {cluster_info.dashboard_address}")

    # --- Test the connection (Optional) ---
    @ray.remote
    def test_remote_function():
        return "Hello from the existing Ray cluster!"

    print(f"Test remote function result: {ray.get(test_remote_function.remote())}")

    # Keep the script running to maintain connection.
    # You'll run your main controller.py from another terminal after this is confirmed.
    print("\nRay client is connected. You can now run other Ray applications (like controller.py) in this terminal or another.")
    print("To disconnect, interrupt this script (Ctrl+C).")
    # Keep process alive to keep connection
    input("Press Enter to disconnect from Ray cluster...")
    ray.shutdown()
    print("Disconnected from Ray cluster.")


except Exception as e:
    print(f"Failed to get info or connect to Ray cluster: {e}")
    print("Please ensure the cluster name and project/location are correct, and the cluster is in a RUNNING state.")