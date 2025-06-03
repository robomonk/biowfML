import ray
import subprocess
import time
import os
import psutil
from google.cloud import storage
import json
import yaml # NEW: Import yaml library

# --- Configuration Loading Function ---
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- GCS Utility Functions (for direct download if gcsfuse isn't used) ---
def download_blob_to_file(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"

# --- GCP Cost Modeling (as defined in your design) ---
CPU_COST_PER_HOUR = 0.031611
RAM_COST_PER_HOUR = 0.004237

def calculate_gcp_cost(duration_seconds, cpu_allocated, ram_allocated_gb):
    cpu_cost_per_sec = (CPU_COST_PER_HOUR / 3600) * cpu_allocated
    ram_cost_per_sec = (RAM_COST_PER_HOUR / 3600) * ram_allocated_gb
    total_cost = duration_seconds * (cpu_cost_per_sec + ram_cost_per_sec)
    return total_cost

# --- Ray Remote Function for Task Execution ---
@ray.remote
def run_computational_task(
    task_id: str,
    task_type: str, # e.g., "bwa_mem", "samtools_sort"
    input_gcs_paths: dict, # e.g., {'reads': 'gs://...'} - these are full GCS URIs
    output_gcs_dir: str, # e.g., 'gs://your-results-bucket/task_outputs/' - full GCS URI
    cpu_allocation: int,
    ram_allocation_gb: int,
    pipeline_config: dict # NEW: Pass the entire loaded config here
) -> dict:
    """
    Executes a genomics computational task using a Docker container,
    referencing paths and tool images from the provided config.
    """
    start_time = time.time()
    exit_status = 1 # Assume failure by default
    output_files = {} # To store paths of generated output files

    local_work_dir = f"/tmp/ray_task_{task_id}"
    os.makedirs(local_work_dir, exist_ok=True)
    print(f"[{task_id}] Task {task_type} started in {local_work_dir}")

    # --- Get Configured Paths & Tool Images ---
    gcs_fuse_mount_base = pipeline_config['gcs_paths']['gcs_fuse_mount_base']
    output_results_bucket_name = pipeline_config['gcs_paths']['output_results_bucket']
    tool_image = pipeline_config['tool_images'].get(task_type)

    if not tool_image:
        raise ValueError(f"Tool image for task type '{task_type}' not found in config.yaml")

    # Helper to convert GCS path to local mounted path
    def get_mounted_path(gcs_path):
        # Example: gs://your-bucket-name/path/to/file.fq -> /mnt/gcs/your-bucket-name/path/to/file.fq
        if gcs_path.startswith("gs://"):
            bucket_name = gcs_path.split('//')[1].split('/')[0]
            blob_path = '/'.join(gcs_path.split('/')[3:])
            return os.path.join(gcs_fuse_mount_base, bucket_name, blob_path)
        return gcs_path # Return as is if not a GCS path

    try:
        # --- Prepare Input Paths for Tool Container ---
        local_input_paths = {}
        for key, gcs_path in input_gcs_paths.items():
            local_input_paths[key] = get_mounted_path(gcs_path)
            if not os.path.exists(local_input_paths[key]):
                print(f"WARNING: Mounted input {local_input_paths[key]} for {key} does not exist! Check GCS_MOUNTS setup.")

        # Determine the actual output directory for the tool container
        # This needs to be a path accessible by the Docker container itself.
        # We assume it writes to the mounted GCS output path.
        local_output_dir_for_tool = get_mounted_path(output_gcs_dir)
        os.makedirs(local_output_dir_for_tool, exist_ok=True)
        print(f"[{task_id}] Tool outputs will go to mounted path: {local_output_dir_for_tool}")

        # --- Define Docker Commands for Genomics Tools ---
        docker_cmd_prefix = ["docker", "run", "--rm"]
        docker_cmd_prefix += [f"--cpus={cpu_allocation}", f"--memory={ram_allocation_gb}g"]

        # Mount the base GCS mount path into the container, if necessary,
        # so the tool container sees /mnt/gcs/your-bucket/ as expected.
        # This assumes the tool container knows about /mnt/gcs/ or similar structure.
        # You might need specific -v mounts for each input/output directory, depending on your tool image.
        # For simplicity, let's assume direct access to mounted paths as defined by get_mounted_path.

        # --- Tool-specific Commands ---
        if task_type == "bwa_mem":
            ref_fa = local_input_paths['ref']
            read1_fq = local_input_paths['read1']
            read2_fq = local_input_paths.get('read2')

            # Full path to the output BAM file in the mounted GCS output directory
            output_bam_mounted_path = os.path.join(local_output_dir_for_tool, f"{task_id}_aligned.bam")

            cmd_parts = [
                tool_image, # Use the image from config
                "mem",
                ref_fa,
                read1_fq
            ]
            if read2_fq:
                cmd_parts.append(read2_fq)

            # Example: BWA outputs to stdout, pipe to samtools view and sort
            # This is more complex to do with separate Docker containers in one subprocess.run.
            # A common pattern is:
            # 1. Run BWA, pipe stdout to a temp file on local_work_dir
            # 2. Run Samtools view on temp file, pipe stdout to another temp file
            # 3. Run Samtools sort on that temp file, output to final BAM.
            # For simplicity, we'll assume the BWA container can write directly to `output_bam_mounted_path`
            # (which requires the container to have write access to that mounted path)

            print(f"[{task_id}] Running BWA mem with image: {tool_image}")
            print(f"[{task_id}] Command: {' '.join(docker_cmd_prefix + cmd_parts)} > {output_bam_mounted_path}")

            # Simulated command for now. REPLACE WITH ACTUAL SUBPROCESS.RUN
            # Example of actual docker run command for a single tool producing an output file:
            # docker run --rm -v /mnt/gcs:/mnt/gcs <your_bwa_image> bwa mem /mnt/gcs/your-refs/transcript.fa /mnt/gcs/your-reads/gut_1.fq > /mnt/gcs/your-results/aligned_bams/task_id_aligned.sam
            # Then use samtools to convert/sort.
            # Or, if your tool image wraps the whole workflow, it might just need input/output paths.

            # For a more direct approach, assuming the tool image has the entrypoint to run bwa directly:
            full_docker_cmd = docker_cmd_prefix + [tool_image] + cmd_parts[1:] # Skip first item (image name) if image is entrypoint
            # If the image *is* the bwa binary, then just use:
            # full_docker_cmd = docker_cmd_prefix + [tool_image] + cmd_parts

            # For RNA-Seq, usually BWA aligns to genome, then STAR to transcriptome.
            # Let's simplify and make a dummy file output.

            # --- SIMULATION OF TOOL EXECUTION ---
            # In a real scenario, you'd run `subprocess.run(full_docker_cmd, ...)`
            # and ensure the output file is created by the container.

            # Create a dummy output file for demonstration
            with open(output_bam_mounted_path, "w") as f:
                f.write(f"Simulated BWA aligned BAM content for {task_id}\n")
            print(f"[{task_id}] Simulated BWA output to {output_bam_mounted_path}")

            output_files['aligned_bam'] = output_bam_mounted_path # This is the mounted path on worker
            exit_status = 0 # Simulate success

        elif task_type == "samtools_sort":
            input_bam = local_input_paths['input_bam']
            output_sorted_bam_mounted_path = os.path.join(local_output_dir_for_tool, f"{task_id}_sorted.bam")

            print(f"[{task_id}] Running Samtools sort with image: {tool_image}")
            # Simulated command. REPLACE WITH ACTUAL SUBPROCESS.RUN
            # Example: docker run --rm -v /mnt/gcs:/mnt/gcs <your_samtools_image> samtools sort -o /mnt/gcs/your-results/aligned_bams/task_id_sorted.bam /mnt/gcs/your-results/aligned_bams/task_id_aligned.bam
            with open(output_sorted_bam_mounted_path, "w") as f: # Create a dummy file
                f.write(f"Simulated Samtools sorted BAM content for {task_id}\n")
            print(f"[{task_id}] Simulated Samtools sort output to {output_sorted_bam_mounted_path}")

            output_files['sorted_bam'] = output_sorted_bam_mounted_path
            exit_status = 0 # Simulate success

        elif task_type == "dummy_task": # Example for testing
            print(f"[{task_id}] Running dummy task with CPU {cpu_allocation} and RAM {ram_allocation_gb}GB...")
            time.sleep(2 + (cpu_allocation * 0.1) + (ram_allocation_gb * 0.05)) # Simulate work
            dummy_output_file = os.path.join(local_output_dir_for_tool, f"{task_id}_dummy_output.txt")
            with open(dummy_output_file, "w") as f:
                f.write(f"This is a dummy output for task {task_id}\n")
            output_files['dummy_output'] = dummy_output_file
            exit_status = 0 # Simulate success

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # --- Capture Resource Utilization (Approximation for the worker node) ---
        # This is a basic approximation. For true container-level utilization,
        # you'd use 'docker stats' or cgroup parsing, which is more complex.
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent(interval=0.1) # % CPU time since last call
        ram_info = process.memory_info()
        # Convert to GB
        ram_utilization_gb = ram_info.rss / (1024**3)

        # A rough estimate based on allocated vs utilized
        # Note: This is imperfect as other Ray processes share the worker.
        avg_cpu_util_ratio = cpu_percent / (psutil.cpu_count() * 100) # Normalize to 0-1 based on total cores
        avg_ram_util_ratio = ram_utilization_gb / ram_allocation_gb if ram_allocation_gb > 0 else 0.0

        # If you want to log actual GCS paths for the output files:
        gcs_output_files = {}
        for key, mounted_path in output_files.items():
            # Convert mounted path back to GCS URI for logging
            # This assumes mounted_path is like /mnt/gcs/bucket_name/blob_path
            relative_path = mounted_path[len(gcs_fuse_mount_base):]
            bucket_name_from_path = relative_path.split('/')[0]
            blob_name_from_path = '/'.join(relative_path.split('/')[1:])
            gcs_output_files[key] = f"gs://{bucket_name_from_path}/{blob_name_from_path}"


    except subprocess.CalledProcessError as e:
        print(f"[{task_id}] Task {task_type} failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout.decode()}")
        print(f"STDERR: {e.stderr.decode()}")
        exit_status = e.returncode
    except Exception as e:
        print(f"[{task_id}] An unexpected error occurred during {task_type}: {e}")
        exit_status = 1

    finally:
        end_time = time.time()
        duration_seconds = end_time - start_time
        task_cost = calculate_gcp_cost(duration_seconds, cpu_allocation, ram_allocation_gb)

        print(f"[{task_id}] Task {task_type} finished. Duration: {duration_seconds:.2f}s, Cost: ${task_cost:.4f}, Status: {'Success' if exit_status == 0 else 'Failure'}")
        # Clean up local work directory if it was used for temp files
        # import shutil
        # shutil.rmtree(local_work_dir, ignore_errors=True)

    return {
        'task_id': task_id,
        'task_type': task_type,
        'duration_seconds': duration_seconds,
        'cost': task_cost,
        'cpu_utilization': avg_cpu_util_ratio, # Normalized 0-1
        'ram_utilization_gb': avg_ram_util_ratio * ram_allocated_gb, # Actual GB used (approx)
        'cpu_allocated': cpu_allocation,
        'ram_allocated_gb': ram_allocated_gb,
        'exit_status': exit_status,
        'success': exit_status == 0,
        'output_files': gcs_output_files # Actual GCS URIs
    }