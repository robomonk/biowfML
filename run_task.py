import ray
import subprocess
import time
import os
# Removed psutil, json, yaml
import re
import logging
from utils import parse_gcs_uri, InvalidGCSPathError # Added

logger = logging.getLogger(__name__)

# load_config function removed

# --- GCP Cost Modeling ---
def calculate_gcp_cost(duration_seconds, cpu_allocated, ram_allocated_gb, cost_config: dict):
    cpu_cost_per_hour = cost_config['cpu_per_hour']
    ram_cost_per_hour = cost_config['ram_per_gb_hour']
    cpu_cost_per_sec = (cpu_cost_per_hour / 3600) * cpu_allocated
    ram_cost_per_sec = (ram_cost_per_hour / 3600) * ram_allocated_gb
    total_cost = duration_seconds * (cpu_cost_per_sec + ram_cost_per_sec)
    return total_cost

def parse_memory_to_gb(mem_str):
    # Parses memory strings like "700.1MiB", "1.5GiB", "100KiB" into GB
    # Returns 0.0 on parsing error or if input is not a string
    if not isinstance(mem_str, str):
        return 0.0
    match = re.match(r'(\d+\.?\d*)\s*([KMGT]?i?B)', mem_str, re.IGNORECASE)
    if not match:
        return 0.0

    val = float(match.group(1))
    unit = match.group(2).upper()

    if unit.startswith('K'):
        return val / (1024**2) # KiB to GB
    elif unit.startswith('M'):
        return val / 1024      # MiB to GB
    elif unit.startswith('G'):
        return val             # GiB to GB
    elif unit.startswith('T'):
        return val * 1024      # TiB to GB
    else: # Bytes
        return val / (1024**3) # B to GB

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
    avg_cpu_util_ratio = 0.0 # Initialize resource metrics
    actual_ram_util_gb = 0.0 # Initialize resource metrics

    local_work_dir = f"/tmp/ray_task_{task_id}" # Temporary local directory on the worker VM
    os.makedirs(local_work_dir, exist_ok=True) # Create if it doesn't exist
    logger.info(f"Task {task_type} (ID: {task_id}) started in {local_work_dir}")

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
            try:
                bucket_name, blob_path = parse_gcs_uri(gcs_path)
            except InvalidGCSPathError: # Should not happen if gcs_path.startswith("gs://") is true
                # This case implies an internal logic error if reached, as starts_with was checked
                logger.error(f"Internal error: get_mounted_path called with invalid GCS URI after check: {gcs_path}")
                return gcs_path # Fallback to old behavior, though problematic
            return os.path.join(gcs_fuse_mount_base, bucket_name, blob_path)
        return gcs_path # Return as is if not a GCS path (e.g., dynamic_from_previous_task)

    # Construct the base docker run command prefix
    # Mount the entire /mnt/gcs to /mnt/gcs inside the container
    # This assumes tools can then directly use paths like /mnt/gcs/<bucket>/<file>
    # And that the tool images are configured to see this mount.
    container_name = f"container-{task_id}"
    docker_cmd_prefix = ["docker", "run", f"--name={container_name}", "-v", f"{gcs_fuse_mount_base}:{gcs_fuse_mount_base}"]
    docker_cmd_prefix += [f"--cpus={cpu_allocation}", f"--memory={ram_allocation_gb}g"]
    # Add the tool image as the first argument after prefix
    # Note: For bwa_mem_align, docker_cmd_with_image is redefined locally later. This is OK.
    docker_cmd_with_image = docker_cmd_prefix + [tool_image]

    # Convert GCS input/output paths to their mounted counterparts
    mounted_inputs = {k: get_mounted_path(v) for k, v in input_gcs_paths.items() if v != 'dynamic_from_previous_task_intervals'}
    mounted_output_dir = get_mounted_path(output_gcs_dir)
    os.makedirs(mounted_output_dir, exist_ok=True) # Ensure output directory exists

    logger.info(f"Task {task_id} Mounted Inputs: {mounted_inputs}")
    logger.info(f"Task {task_id} Mounted Output Dir: {mounted_output_dir}")

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

            # Define the docker command for BWA specifically, including the container name
            # This reuses docker_cmd_prefix which now includes the container name
            bwa_docker_cmd_prefix = docker_cmd_prefix # docker_cmd_prefix already has --name

            cmd_for_bwa = [
                "mem", "-R", "@RG\\tID:t1\\tSM:t1", # Use raw string for @RG to avoid python interpretation
                ref_file, read1
            ]
            if read2:
                cmd_for_bwa.append(read2)

            # Execute BWA, capture its stdout
            # The docker_cmd_with_image here is specific for BWA, including the image and then the bwa command
            final_bwa_docker_command = bwa_docker_cmd_prefix + [tool_image] + cmd_for_bwa
            logger.info(f"Task {task_id} Running BWA mem with command: {' '.join(final_bwa_docker_command)}")
            bwa_process = subprocess.run(
                final_bwa_docker_command,
                check=True, # Will raise CalledProcessError on non-zero exit
                capture_output=True,
                text=True # Decode stdout/stderr as text
            )

            # Write BWA's stdout to the specified output SAM file
            with open(output_sam_file, "w") as f:
                f.write(bwa_process.stdout)

            output_files['aligned_sam'] = output_sam_file # GCS path will be handled at return
            # exit_status will be set after stats collection attempt based on bwa_process.returncode
            # For now, if we are here, bwa_process was successful due to check=True


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
        # --- Execute the Docker Command (for non-BWA tasks) ---
        if task_type != "bwa_mem_align": # BWA already ran
            final_docker_command = docker_cmd_with_image + tool_args
            logger.info(f"Task {task_id} Executing: {' '.join(final_docker_command)}")
            process = subprocess.run(final_docker_command, check=True, capture_output=True, text=True)

            if process.stdout:
                logger.debug(f"Task {task_id} STDOUT:\n{process.stdout[:2000]}...") # DEBUG level for potentially verbose output
            if process.stderr:
                logger.debug(f"Task {task_id} STDERR:\n{process.stderr[:2000]}...") # DEBUG level
            # exit_status will be set after stats collection attempt based on process.returncode

        # --- Collect Docker Stats ---
        # This block runs if the main docker command was successful (or even if not, stats might be available)
        # For bwa_mem_align, bwa_process holds the result. For others, process holds it.
        # We assume if check=True didn't fail, the process object exists and returncode is 0.
        # If an exception occurred before process/bwa_process was set (e.g. config error), this won't run.

        # Determine task success before stats collection attempt
        if task_type == "bwa_mem_align":
            if 'bwa_process' in locals() and bwa_process.returncode == 0:
                exit_status = 0
            elif 'bwa_process' in locals():
                exit_status = bwa_process.returncode
            else: # bwa_process not even defined, means major issue before execution
                exit_status = 1 # Default to failure
        else: # For other tasks
            if 'process' in locals() and process.returncode == 0:
                exit_status = 0
            elif 'process' in locals():
                exit_status = process.returncode
            else: # process not even defined
                exit_status = 1 # Default to failure

        try:
            stats_cmd = ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}} / {{.MemUsage}}", container_name]
            logger.info(f"Task {task_id} Getting stats with: {' '.join(stats_cmd)}")
            stats_process = subprocess.run(stats_cmd, capture_output=True, text=True, check=False)

            if stats_process.returncode == 0 and stats_process.stdout.strip():
                logger.info(f"Task {task_id} Docker stats output: {stats_process.stdout.strip()}")
                parts = stats_process.stdout.strip().split('/')
                cpu_perc_str = parts[0].strip().replace('%', '')
                # MemUsage part can be "700.1MiB / 1.952GiB", so we take the first part before " / "
                mem_usage_str = parts[1].strip().split(' ')[0]


                try:
                    cpu_perc_val = float(cpu_perc_str)
                    avg_cpu_util_ratio = (cpu_perc_val / 100.0) / cpu_allocation
                    avg_cpu_util_ratio = max(0.0, min(avg_cpu_util_ratio, 1.0))
                except ValueError:
                    logger.warning(f"Task {task_id} Warning: Could not parse CPU percentage from stats: {cpu_perc_str}")
                    avg_cpu_util_ratio = 0.0

                actual_ram_util_gb = parse_memory_to_gb(mem_usage_str)
                logger.info(f"Task {task_id} Parsed Stats - CPU Util Ratio (rel. to alloc): {avg_cpu_util_ratio:.2f}, RAM Usage: {actual_ram_util_gb:.2f} GB")

            else:
                logger.warning(f"Task {task_id} Warning: Could not retrieve Docker stats. stdout: {stats_process.stdout}, stderr: {stats_process.stderr}")
                # Defaults avg_cpu_util_ratio = 0.0, actual_ram_util_gb = 0.0 already set

        except Exception as e_stats:
            logger.exception(f"Task {task_id} Error collecting Docker stats:")
            # Defaults avg_cpu_util_ratio = 0.0, actual_ram_util_gb = 0.0 already set

    except subprocess.CalledProcessError as e:
        logger.error(f"Task {task_type} (ID: {task_id}) failed with exit code {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout: logger.error(f"Task {task_id} STDOUT: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr: logger.error(f"Task {task_id} STDERR: {e.stderr}")
        exit_status = e.returncode
    except Exception as e:
        logger.exception(f"Task {task_id} An unexpected error occurred during {task_type}:")
        exit_status = 1 # General failure

    finally:
        # Attempt to remove the container
        logger.info(f"Task {task_id} Attempting to remove container {container_name}...")
        try:
            # Use capture_output and text=True to avoid printing to console unless debugging
            rm_stats = subprocess.run(["docker", "rm", container_name], check=False, capture_output=True, text=True)
            if rm_stats.returncode == 0:
                logger.info(f"Task {task_id} Container {container_name} removed successfully.")
            else:
                # It's common for rm to fail if the container never started, so only log if verbose/debug needed
                logger.info(f"Task {task_id} Info: Could not remove container {container_name} (it might have already been removed or failed to start). stderr: {rm_stats.stderr.strip()}")
        except Exception as e_rm:
            logger.warning(f"Task {task_id} Warning: Exception during attempt to remove container {container_name}: {e_rm}")

        end_time = time.time()
        duration_seconds = end_time - start_time
        task_cost = calculate_gcp_cost(duration_seconds, cpu_allocation, ram_allocation_gb, pipeline_config['gcp_costs'])

        logger.info(f"Task {task_type} (ID: {task_id}) finished. Duration: {duration_seconds:.2f}s, Cost: ${task_cost:.4f}, Status: {'Success' if exit_status == 0 else 'Failure'}")
        # Clean up local work directory if it was used for temp files.
        # For gcsfuse, temp files might not be heavily used, but it's good practice.
        # import shutil
        # shutil.rmtree(local_work_dir, ignore_errors=True)

    # Convert mounted output paths back to GCS URIs for logging
    gcs_output_files = {}
    for key, mounted_path in output_files.items():
        if mounted_path.startswith(gcs_fuse_mount_base):
                # Correctly reconstruct GCS path from mounted path
                # Example: /mnt/gcs_fuse/my-bucket/path/to/file.txt -> gs://my-bucket/path/to/file.txt
                path_without_base = mounted_path[len(gcs_fuse_mount_base):].lstrip('/')
                bucket_name_from_path = path_without_base.split('/')[0]
                blob_name_from_path = '/'.join(path_without_base.split('/')[1:])
                gcs_output_files[key] = f"gs://{bucket_name_from_path}/{blob_name_from_path}"
        else:
            gcs_output_files[key] = mounted_path # If it was already a GCS path or not mounted correctly


    return {
        'task_id': task_id,
        'task_type': task_type,
        'duration_seconds': duration_seconds,
        'cost': task_cost,
        'cpu_utilization': avg_cpu_util_ratio, # Updated: Normalized 0-1 from docker stats
        'ram_utilization_gb': actual_ram_util_gb, # Updated: Actual GB used from docker stats
        'cpu_allocated': cpu_allocation,
        'ram_allocated_gb': ram_allocation_gb,
        'exit_status': exit_status,
        'success': exit_status == 0,
        'output_files': gcs_output_files # Actual GCS URIs of main outputs
    }