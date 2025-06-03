import ray
from google.cloud import aiplatform_v1beta1 as aiplatform
from vertex_ray import Resources, create_ray_cluster, get_ray_cluster, delete_ray_cluster
import time
import os

# --- Configuration ---
PROJECT_ID = "rsabawi-hpc-sandbox-926388"
LOCATION = "us-central1"
CLUSTER_NAME = "genomics-ray-cluster" # Note: User's cluster name is bioflow-20250603-002657
                                      # This script creates a new one, so this can be a different name.
                                      # Or, if it's meant to recreate the existing one, update this.
SERVICE_ACCOUNT_EMAIL = "service-263885197854@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"
# GCS path where you uploaded your Python scripts and gcsfuse_startup.sh
CODE_GCS_PATH = f"gs://{PROJECT_ID}-genomics-results/code/"

# --- 1. Define Ray Cluster Resources ---
head_node_type = Resources(
    machine_type="n1-standard-4", # 4 vCPUs, 16GB RAM for head
    node_count=1,
)

worker_node_types = [
    Resources(
        machine_type="n1-standard-8", # 8 vCPUs, 32GB RAM for workers
        min_replica_count=1,
        max_replica_count=3, # Start with a small max for cost control
    )
]

# --- 2. Create the Ray Cluster ---
print(f"Creating Ray cluster '{CLUSTER_NAME}' in {LOCATION} for project {PROJECT_ID}...")

ray_cluster_name = None # Initialize to None for cleanup in case of partial creation
try:
    ray_cluster_name = create_ray_cluster(
        head_node_type=head_node_type,
        worker_node_types=worker_node_types,
        cluster_name=CLUSTER_NAME,
        project_id=PROJECT_ID,
        location=LOCATION,
        service_account=SERVICE_ACCOUNT_EMAIL,
        startup_script_uris=[os.path.join(CODE_GCS_PATH, "gcsfuse_startup.sh")],
        requirements_uris=[os.path.join(CODE_GCS_PATH, "requirements.txt")],
        working_dir=CODE_GCS_PATH,
    )
    print(f"Ray cluster creation initiated: {ray_cluster_name}. This may take 5-10 minutes.")

    # --- 3. Wait for Cluster to be Ready and Connect ---
    status = None
    while status != "RUNNING":
        cluster_info = get_ray_cluster(
            project_id=PROJECT_ID,
            location=LOCATION,
            cluster_name=CLUSTER_NAME
        )
        status = cluster_info.state.name
        print(f"Cluster status: {status}...")
        if status == "ERROR":
            raise Exception(f"Ray cluster failed to start. Details: {cluster_info.error}")
        if status != "RUNNING":
            time.sleep(30) # Wait 30 seconds before checking again

    print(f"Ray cluster '{CLUSTER_NAME}' is RUNNING. Connecting...")

    # Initialize Ray client to connect to the newly created cluster
    ray.init(
        address=f"vertex_ray://{ray_cluster_name}",
        runtime_env={
            "working_dir": CODE_GCS_PATH,
        }
    )
    print("Successfully connected to Ray cluster.")
    print(f"Ray Dashboard: {cluster_info.dashboard_address}")

except Exception as e:
    print(f"Failed to create or connect to Ray cluster: {e}")
    if ray_cluster_name:
        print(f"Attempting to delete partially created/failed cluster: {ray_cluster_name}")
        try:
            delete_ray_cluster(project_id=PROJECT_ID, location=LOCATION, cluster_name=CLUSTER_NAME)
            print("Cluster deleted.")
        except Exception as delete_e:
            print(f"Failed to delete cluster: {delete_e}")