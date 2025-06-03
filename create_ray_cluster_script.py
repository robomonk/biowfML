import ray
from google.cloud import aiplatform_v1beta1 as aiplatform
from vertex_ray import Resources, create_ray_cluster, get_ray_cluster, delete_ray_cluster
import time
import os
import argparse
import yaml
from google.cloud import storage # Added for GCS upload
import logging # Added logging

# --- Configuration via Arguments ---
parser = argparse.ArgumentParser(description="Create a new Vertex AI Ray cluster.")
parser.add_argument("--project-id", required=True, help="Your GCP Project ID")
parser.add_argument("--location", default="us-central1", help="GCP region for the cluster")
parser.add_argument("--cluster-name", default="genomics-ray-cluster", help="Name for the new Ray cluster")
parser.add_argument("--service-account-email", required=True, help="Service account email for the Ray cluster")
parser.add_argument("--code-gcs-path", required=True, help="GCS path to the code directory (e.g., gs://bucket/code/)")
parser.add_argument("--config-file", default="config.yaml", help="Path to the pipeline config YAML file to extract GCS bucket names for gcsfuse.")
args = parser.parse_args()

PROJECT_ID = args.project_id
LOCATION = args.location
CLUSTER_NAME = args.cluster_name
SERVICE_ACCOUNT_EMAIL = args.service_account_email
CODE_GCS_PATH = args.code_gcs_path
CONFIG_FILE = args.config_file

# --- Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Dynamically Generate gcsfuse_startup.sh ---
logger.info(f"Loading pipeline configuration from: {CONFIG_FILE}")
with open(CONFIG_FILE, 'r') as f:
    pipeline_config = yaml.safe_load(f)

gcs_buckets_to_mount = pipeline_config.get('gcs_fuse_mounts', {}).get('buckets_to_mount', [])

# Add buckets from gcs_paths
if 'gcs_paths' in pipeline_config:
    for key, value in pipeline_config['gcs_paths'].items():
        if isinstance(value, str) and value.startswith("gs://"):
            bucket_name = value.split('//')[1].split('/')[0]
            if bucket_name not in gcs_buckets_to_mount:
                gcs_buckets_to_mount.append(bucket_name)
        elif key.endswith("_bucket") and isinstance(value, str) and value not in gcs_buckets_to_mount:
             # Handles cases where gcs_paths might just have bucket names directly
             gcs_buckets_to_mount.append(value)

gcs_buckets_to_mount = sorted(list(set(gcs_buckets_to_mount))) # Unique, sorted
logger.info(f"Buckets to FUSE mount: {gcs_buckets_to_mount}")

bucket_list_str = "\n".join([f"    \"{b}\"" for b in gcs_buckets_to_mount]) # Format for bash array

template_script_path = "gcsfuse_startup.sh.template"
# Assuming template is in the same directory or accessible via relative path.
# If this script is run from a location where the template isn't, this path needs adjustment
# or the template needs to be fetched from GCS if CODE_GCS_PATH is the source of truth for code.
# For now, assuming local accessibility.

if not os.path.exists(template_script_path):
    # Attempt to find it relative to this script's location if not in CWD
    script_dir = os.path.dirname(os.path.realpath(__file__))
    potential_template_path = os.path.join(script_dir, template_script_path)
    if os.path.exists(potential_template_path):
        template_script_path = potential_template_path
    else:
        # As a last resort, if CODE_GCS_PATH is provided and seems to be the source of other code,
        # one might try to download it. However, this script is for CREATING the cluster,
        # so gcsfuse might not be running yet. Simplest is to ensure template is with the script.
        raise FileNotFoundError(f"gcsfuse_startup.sh.template not found at {template_script_path} or relative locations.")


with open(template_script_path, 'r') as f_template:
    startup_script_content = f_template.read()

startup_script_content = startup_script_content.replace("GCS_BUCKETS_TO_MOUNT_PLACEHOLDER", bucket_list_str)

generated_script_name = "gcsfuse_generated_startup.sh"
generated_script_local_path = os.path.join(os.getcwd(), generated_script_name)
with open(generated_script_local_path, 'w') as f_generated:
    f_generated.write(startup_script_content)

# Upload the generated script to CODE_GCS_PATH
storage_client = storage.Client(project=PROJECT_ID)
# Ensure CODE_GCS_PATH is just the path, not gs://bucket/path
if not CODE_GCS_PATH.startswith("gs://"):
    raise ValueError("CODE_GCS_PATH must be a GCS URI (e.g. gs://bucket/path/)")

code_bucket_name = CODE_GCS_PATH.split('//')[1].split('/')[0]
# Path within the bucket, ensure it ends with / if it's a directory prefix
code_path_prefix = '/'.join(CODE_GCS_PATH.split('//')[1].split('/')[1:])
if code_path_prefix and not code_path_prefix.endswith('/'):
    code_path_prefix += '/'

blob_path_in_bucket = os.path.join(code_path_prefix, generated_script_name)

bucket_obj = storage_client.bucket(code_bucket_name)
blob_obj = bucket_obj.blob(blob_path_in_bucket)
blob_obj.upload_from_filename(generated_script_local_path)
startup_script_final_uri = f"gs://{code_bucket_name}/{blob_path_in_bucket}"
logger.info(f"Uploaded generated startup script to {startup_script_final_uri}")

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
logger.info(f"Creating Ray cluster '{CLUSTER_NAME}' in {LOCATION} for project {PROJECT_ID}...")

ray_cluster_name = None # Initialize to None for cleanup in case of partial creation
try:
    ray_cluster_name = create_ray_cluster(
        head_node_type=head_node_type,
        worker_node_types=worker_node_types,
        cluster_name=CLUSTER_NAME,
        project_id=PROJECT_ID,
        location=LOCATION,
        service_account=SERVICE_ACCOUNT_EMAIL,
        startup_script_uris=[startup_script_final_uri], # Use the generated and uploaded script
        requirements_uris=[os.path.join(CODE_GCS_PATH, "requirements.txt")], # Assuming requirements.txt is also in CODE_GCS_PATH
        working_dir=CODE_GCS_PATH,
    )
    logger.info(f"Ray cluster creation initiated: {ray_cluster_name}. This may take 5-10 minutes.")

    # --- 3. Wait for Cluster to be Ready and Connect ---
    status = None
    while status != "RUNNING":
        cluster_info = get_ray_cluster(
            project_id=PROJECT_ID,
            location=LOCATION,
            cluster_name=CLUSTER_NAME
        )
        status = cluster_info.state.name
        logger.info(f"Cluster status: {status}...")
        if status == "ERROR":
            logger.error(f"Ray cluster failed to start. Details: {cluster_info.error}")
            raise Exception(f"Ray cluster failed to start. Details: {cluster_info.error}")
        if status != "RUNNING":
            time.sleep(30) # Wait 30 seconds before checking again

    logger.info(f"Ray cluster '{CLUSTER_NAME}' is RUNNING. Connecting...")

    # Initialize Ray client to connect to the newly created cluster
    ray.init(
        address=f"vertex_ray://{ray_cluster_name}",
        runtime_env={
            "working_dir": CODE_GCS_PATH,
        }
    )
    logger.info("Successfully connected to Ray cluster.")
    logger.info(f"Ray Dashboard: {cluster_info.dashboard_address}")

except Exception as e:
    logger.exception("Failed to create or connect to Ray cluster:")
    if ray_cluster_name:
        logger.warning(f"Attempting to delete partially created/failed cluster: {ray_cluster_name}")
        try:
            delete_ray_cluster(project_id=PROJECT_ID, location=LOCATION, cluster_name=CLUSTER_NAME)
            logger.info("Cluster deleted.")
        except Exception as delete_e:
            logger.exception(f"Failed to delete cluster {ray_cluster_name}:")