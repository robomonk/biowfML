import ray
import subprocess
import time
import os
import psutil
from google.cloud import storage
import json
import yaml

# --- Configuration Loading Function (kept for completeness, but config is passed in) ---
def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- GCS Utility Functions (for direct download if gcsfuse isn't used) ---
# These functions are here if you opt not to use gcsfuse.
def download_blob_to_file(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def upload_file_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage_client.bucket(bucket_name)
    blob = storage_client.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")
    return f"gs://{bucket_name}/{destination_blob_name}"

# --- GCP Cost Modeling ---
# These are illustrative values, adjust based on actual GCP pricing for your region
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
    task_type: str, # e.g., "bwa_index", "picard_sort_sam"
    input_gcs_paths: dict, # e.g., {'fasta': 'gs://...'} - these are full GCS URIs
    output_gcs_dir: str, # e.g., 'gs://your-results-bucket/task_outputs/' - full GCS URI
    cpu_allocation: int,
    ram_allocation_gb: int,
    pipeline_config: dict # Pass the entire loaded config here
) -> dict:
    """
    Executes a genomics computational task using a Docker container,
    referencing paths and tool images from the provided config.
    """
    start_time = time.time()
    exit_status = 1 # Assume failure by default
    output_files = {} # To store GCS paths of generated output files

    local_work_dir = f"/tmp/ray_task_{task_id}" # Temporary local directory on the worker VM
    os.makedirs(local_work_dir, exist_ok=True) # Create if it doesn't exist
    print(f"[{task_id}] Task {task_type} started in {local_work_dir}")

    # --- Get Configured Paths & Tool Images ---
    gcs_fuse_mount_base = pipeline_config['gcs_paths']['gcs_fuse_mount_base']
    tool_image_base_name = task_type.split('_')[0] # e.g., 'bwa' from 'bwa_index'
    tool_image = pipeline_config['tool_images'].get(tool_image_base_name)

    if not tool_image:
        raise ValueError(f"Tool image for task type '{task_type}' (base: {tool_image_base_name}) not found in config.yaml under 'tool_images'.")

    # Helper to convert GCS path to local mounted path
    # Assumes GCS buckets are mounted under MOUNT_BASE/<bucket_name>/
    def get_mounted_path(gcs_path):
        if gcs_path.startswith("gs://"):
            bucket_name = gcs_path.split('//')[1].split('/')[0]
            blob_path = '/'.join(gcs_path.split('/')[3:])
            return os.path.join(gcs_fuse_mount_base, bucket_name, blob_path)
        return gcs_path # Return as is if not a GCS path (e.g., dynamic_from_previous_task)

    # Construct the base docker run command prefix
    # Mount the entire /mnt/gcs to /mnt/gcs inside the container
    # This assumes tools can then directly use paths like /mnt/gcs/<bucket>/<file>
    # And that the tool images are configured to see this mount.
    docker_cmd_prefix = ["docker", "run", "--rm", "-v", f"{gcs_fuse_mount_base}:{gcs_fuse_mount_base}"]
    docker_cmd_prefix += [f"--cpus={cpu_allocation}", f"--memory={ram_allocation_gb}g"]
    # Add the tool image as the first argument after prefix
    docker_cmd_with_image = docker_cmd_prefix + [tool_image]

    # Convert GCS input/output paths to their mounted counterparts
    mounted_inputs = {k: get_mounted_path(v) for k, v in input_gcs_paths.items() if v != 'dynamic_from_previous_task_intervals'}
    mounted_output_dir = get_mounted_path(output_gcs_dir)
    os.makedirs(mounted_output_dir, exist_ok=True) # Ensure output directory exists

    print(f"[{task_id}] Mounted Inputs: {mounted_inputs}")
    print(f"[{task_id}] Mounted Output Dir: {mounted_output_dir}")

    try:
        # --- Tool-specific Commands ---
        tool_args = [] # This will hold the arguments *for the tool itself* (after the image name)
        command_output_to_stdout = False # Flag if tool writes to stdout and needs redirection

        if task_type == "bwa_index":
            fasta_file = mounted_inputs['fasta']
            tool_args = ["index", fasta_file]
            # BWA index creates multiple files in the same directory as input fasta.
            # These are implicitly managed by gcsfuse and will appear in GCS.
            # We just need to record the directory.
            output_files['indexed_ref_dir'] = os.path.dirname(fasta_file) # GCS path will be handled at return

        elif task_type == "samtools_faidx":
            fasta_file = mounted_inputs['fasta']
            tool_args = ["faidx", fasta_file]
            output_fai_file = fasta_file + ".fai"
            output_files['fai_index'] = output_fai_file # GCS path will be handled at return

        elif task_type == "picard_create_dict":
            fasta_file = mounted_inputs['fasta']
            output_dict_file = os.path.splitext(fasta_file)[0] + ".dict"
            # For Picard, the general pattern is 'picard <ToolName> ARGS'
            tool_args = ["CreateSequenceDictionary", f"R={fasta_file}", f"O={output_dict_file}"]
            output_files['dict_index'] = output_dict_file # GCS path will be handled at return

        elif task_type == "bwa_mem_align":
            ref_file = mounted_inputs['ref']
            read1 = mounted_inputs['read1']
            read2 = mounted_inputs.get('read2') # Optional
            output_sam_file = os.path.join(mounted_output_dir, f"{task_id}.sam")

            # BWA mem outputs to stdout, which needs to be redirected to a file.
            # We'll use a temporary local file, then stream its content to the GCS mounted output.
            # Or, if we assume the tool container has direct write access to mounted_output_dir
            # and can handle redirection via bash shell directly.
            # For simplicity and directness, let's capture stdout and write it.
            # The user's example showed `> 7859_GPI.aln_pe.sam` which means shell redirection.
            # So we run the command and capture stdout.

            cmd_for_bwa = [
                "mem", "-R", "@RG\\tID:t1\\tSM:t1", # Use raw string for @RG to avoid python interpretation
                ref_file, read1
            ]
            if read2:
                cmd_for_bwa.append(read2)

            # Execute BWA, capture its stdout
            print(f"[{task_id}] Running BWA mem with command: {' '.join(docker_cmd_with_image + cmd_for_bwa)}")
            bwa_process = subprocess.run(
                docker_cmd_with_image + cmd_for_bwa,
                check=True,
                capture_output=True,
                text=True # Decode stdout/stderr as text
            )

            # Write BWA's stdout to the specified output SAM file
            with open(output_sam_file, "w") as f:
                f.write(bwa_process.stdout)

            output_files['aligned_sam'] = output_sam_file # GCS path will be handled at return
            exit_status = 0 # Assume success if subprocess.run passed check=True


        elif task_type == "picard_sort_sam":
            input_sam = mounted_inputs['input_sam']
            output_bam_file = os.path.join(mounted_output_dir, f"{task_id}_sorted.bam")
            tool_args = ["SortSam", f"INPUT={input_sam}", f"OUTPUT={output_bam_file}", "SORT_ORDER=coordinate"]
            output_files['sorted_bam'] = output_bam_file # GCS path will be handled at return

        elif task_type == "picard_build_bam_idx": # Used for both initial and marked BAMs
            input_bam = mounted_inputs['input_bam']
            output_bai_file = input_bam + ".bai" # BAM index in same dir as BAM
            tool_args = ["BuildBamIndex", "VALIDATION_STRINGENCY=LENIENT", f"I={input_bam}"]
            output_files['bam_index'] = output_bai_file # GCS path will be handled at return

        elif task_type == "picard_mark_duplicates":
            input_bam = mounted_inputs['input_bam']
            output_md_bam_file = os.path.join(mounted_output_dir, f"{task_id}_md.bam")
            output_metrics_file = os.path.join(mounted_output_dir, f"{task_id}_md.metrics")
            # REMOVE_DUPLICATES=true from tutorial, but often set to false to keep reads for downstream analysis
            tool_args = [
                "MarkDuplicates", "VALIDATION_STRINGENCY=LENIENT", "AS=true", "REMOVE_DUPLICATES=true",
                f"I={input_bam}", f"O={output_md_bam_file}", f"M={output_metrics_file}"
            ]
            output_files['marked_duplicates_bam'] = output_md_bam_file # GCS path will be handled at return
            output_files['marked_duplicates_metrics'] = output_metrics_file # GCS path will be handled at return

        elif task_type == "gatk_realign_target":
            ref_file = mounted_inputs['ref']
            input_bam = mounted_inputs['input_bam']
            output_intervals_file = os.path.join(mounted_output_dir, f"{task_id}.intervals")
            # GATK specific command, assuming 'java -jar /path/to/gatk.jar' is how the container runs GATK.
            # Biocontainers might have `gatk` as a symlink or entrypoint, but the tutorial used `java -jar GenomeAnalysisTK.jar`.
            # Assuming /gatk/GenomeAnalysisTK.jar is the path inside the container.
            tool_args = [
                "java", "-jar", "/gatk/GenomeAnalysisTK.jar", "-T", "RealignerTargetCreator",
                "-nt", str(cpu_allocation), "-R", ref_file, "-I", input_bam, "-o", output_intervals_file
            ]
            output_files['realign_intervals'] = output_intervals_file # GCS path will be handled at return

        elif task_type == "gatk_indel_realign":
            ref_file = mounted_inputs['ref']
            input_bam = mounted_inputs['input_bam']
            intervals_file = mounted_inputs['intervals'] # This will be the mounted path of the intervals file
            output_realigned_bam_file = os.path.join(mounted_output_dir, f"{task_id}_realigned.bam")
            tool_args = [
                "java", "-jar", "/gatk/GenomeAnalysisTK.jar", "-T", "IndelRealigner",
                "-R", ref_file, "-I", input_bam, "-targetIntervals", intervals_file,
                "-o", output_realigned_bam_file
            ]
            output_files['realigned_bam'] = output_realigned_bam_file # GCS path will be handled at return

        elif task_type == "gatk_haplotype_caller":
            ref_file = mounted_inputs['ref']
            input_bam = mounted_inputs['input_bam']
            output_gvcf_file = os.path.join(mounted_output_dir, f"{task_id}.g.vcf")
            tool_args = [
                "java", "-jar", "/gatk/GenomeAnalysisTK.jar", "-T", "HaplotypeCaller",
                "-R", ref_file, "-ERC", "GVCF", "-I", input_bam, "-o", output_gvcf_file
            ]
            output_files['haplotype_gvcf'] = output_gvcf_file # GCS path will be handled at return

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # --- Execute the Docker Command ---
        # Construct the full docker command to run the tool.
        # This is `docker run <prefix> <image> <tool_args>`
        final_docker_command = docker_cmd_with_image + tool_args

        print(f"[{task_id}] Executing: {' '.join(final_docker_command)}")
        process = subprocess.run(final_docker_command, check=True, capture_output=True, text=True) # Capture stdout/stderr

        # Log stdout/stderr if useful for debugging (limit output for brevity)
        if process.stdout:
            print(f"[{task_id}] STDOUT:\n{process.stdout[:2000]}...")
        if process.stderr:
            print(f"[{task_id}] STDERR:\n{process.stderr[:2000]}...")

        exit_status = 0 # Set to 0 if subprocess.run didn't raise CalledProcessError

    except subprocess.CalledProcessError as e:
        print(f"[{task_id}] Task {task_type} failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        exit_status = e.returncode
    except Exception as e:
        print(f"[{task_id}] An unexpected error occurred during {task_type}: {e}")
        exit_status = 1

    finally:
        end_time = time.time()
        duration_seconds = end_time - start_time
        task_cost = calculate_gcp_cost(duration_seconds, cpu_allocation, ram_allocation_gb)

        print(f"[{task_id}] Task {task_type} finished. Duration: {duration_seconds:.2f}s, Cost: ${task_cost:.4f}, Status: {'Success' if exit_status == 0 else 'Failure'}")
        # Clean up local work directory if it was used for temp files.
        # For gcsfuse, temp files might not be heavily used, but it's good practice.
        # import shutil
        # shutil.rmtree(local_work_dir, ignore_errors=True)

    # Convert mounted output paths back to GCS URIs for logging
    gcs_output_files = {}
    for key, mounted_path in output_files.items():
        if mounted_path.startswith(gcs_fuse_mount_base):
            relative_path = mounted_path[len(gcs_fuse_mount_base):]
            bucket_name_from_path = relative_path.split('/')[0]
            blob_name_from_path = '/'.join(relative_path.split('/')[1:])
            gcs_output_files[key] = f"gs://{bucket_name_from_path}/{blob_name_from_path}"
        else:
            gcs_output_files[key] = mounted_path # If it was already a GCS path or not mounted correctly

    # These are dummy values for now. Update this based on actual resource monitoring
    # (e.g., using docker stats for the specific container).
    avg_cpu_util_ratio = 0.5
    avg_ram_util_ratio = 0.5
    try:
        process_info = psutil.Process(os.getpid())
        avg_cpu_util_ratio = process_info.cpu_percent(interval=0.1) / 100.0 / psutil.cpu_count() # Normalize to 0-1 based on physical cores
        ram_utilization_gb = process_info.memory_info().rss / (1024**3)
        avg_ram_util_ratio = ram_utilization_gb / ram_allocation_gb if ram_allocation_gb > 0 else 0.0
    except Exception as e:
        print(f"Warning: Could not get precise psutil metrics: {e}")

    return {
        'task_id': task_id,
        'task_type': task_type,
        'duration_seconds': duration_seconds,
        'cost': task_cost,
        'cpu_utilization': avg_cpu_util_ratio, # Normalized 0-1
        'ram_utilization_gb': avg_ram_util_ratio * ram_allocation_gb, # Actual GB used (approx)
        'cpu_allocated': cpu_allocation,
        'ram_allocated_gb': ram_allocation_gb,
        'exit_status': exit_status,
        'success': exit_status == 0,
        'output_files': gcs_output_files # Actual GCS URIs of main outputs
    }