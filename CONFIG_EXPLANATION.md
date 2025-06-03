# Understanding `config.yaml`

## 1. Overall Purpose

The `config.yaml` file is the central configuration hub for the Reinforcement Learning (RL) agent and the genomics workflows it manages. It dictates where data is stored, what tools are used, how workflows are structured, and how the RL agent should behave and learn. Modifying this file correctly is crucial for adapting the system to your specific datasets, tools, and research goals.

## 2. `gcs_paths`

This section defines key Google Cloud Storage (GCS) locations used by the system for various outputs and operational purposes.

*   **Role:** Specifies base GCS URIs for storing results, logs, intermediate data, and the local mount point for GCS FUSE.
*   **Key sub-fields:**
    *   `output_results_bucket`: The name of the GCS bucket where final workflow results (e.g., aligned BAM files, variant call files) and summary metrics will be stored.
    *   `tensorboard_log_bucket`: The name of the GCS bucket used to store TensorBoard logs generated during the RL agent's training. You can point TensorBoard to this location (e.g., `gs://your-bucket/tensorboard_logs/`) to visualize training progress.
    *   `staging_bucket`: The name of a GCS bucket dedicated to storing intermediate data. For example, if your primary input files are from public GCS buckets, they are first copied (staged) to this bucket to ensure consistent access and ownership during the workflow execution. Each run of the workflow might create a unique subdirectory here.
    *   `gcs_fuse_mount_base`: This is a **local filesystem path** (e.g., `/mnt/gcs/`) on each Ray worker node. GCS FUSE will mount the GCS buckets under this base path. For instance, if `gcs_fuse_mount_base` is `/mnt/gcs/` and you access `gs://my-data-bucket/file.txt`, the tools running in Docker containers on worker nodes will see it as `/mnt/gcs/my-data-bucket/file.txt`.

## 3. `input_sources`

This section lists the GCS URIs for the raw input data files required by your genomics workflows.

*   **Role:** Provides a centralized mapping of logical input names (e.g., `rnaseq_chicken_ref`) to their actual GCS locations (e.g., `gs://rnaseq-nf/data/ggal/transcript.fa`).
*   **How it's used:**
    *   The `WorkflowEnv` uses these paths during the data staging phase to copy necessary input files into the `staging_bucket` for the current workflow run.
    *   The `workflow_definition` section refers to these logical names (e.g., `${input_sources.rnaseq_chicken_ref}`) to specify inputs for particular tasks. This makes it easier to switch datasets without altering the core workflow logic.

## 4. `tool_images`

This section maps abstract tool names to specific Docker image URIs. These images contain the genomics software needed for each pipeline step.

*   **Role:** Decouples the workflow definition from hardcoded Docker image paths.
*   **Key sub-fields:** Each key is an abstract tool name (e.g., `bwa`, `samtools`, `picard`, `gatk`), and its value is the full URI of the Docker image (e.g., `biocontainers/bwa:0.7.17--he4a0461_8`, `broadinstitute/picard:latest`). These can be from Docker Hub, Google Container Registry (GCR), or other container registries.
*   **How it's used:** The `run_task.py` script (specifically, the `run_computational_task` function) uses the `type` field from a task in the `workflow_definition` (e.g., `bwa_mem_align` might map to the `bwa` tool key) to look up the corresponding Docker image URI in this section. It then uses this URI to pull and run the Docker container for that task.

## 5. `workflow_definition`

This is a core section that defines the sequence of computational steps (tasks) in your genomics pipeline.

*   **Role:** Describes the directed acyclic graph (DAG) of your workflow.
*   **Structure:** It's an array (list) of task objects. Each object represents a step in the pipeline.
*   **Key fields per task object:**
    *   `task_id_prefix`: A string used as a base to generate unique identifiers for instances of this task during a workflow run (e.g., `bwa-align`).
    *   `type`: A string that specifies the kind of operation to be performed (e.g., `bwa_mem_align`, `samtools_sort`, `picard_create_dict`). The `run_task.py` script contains logic to interpret this type and execute the appropriate commands within the Docker container.
    *   `inputs`: A dictionary specifying the input files required for this task.
        *   Values can be references to the `input_sources` section using a template format (e.g., `${input_sources.rnaseq_chicken_ref}`). This tells the system to use the (staged) data from that source.
        *   A special value, `"dynamic_from_previous_task"`, indicates that the input for this key will be the primary output of the immediately preceding task in the workflow.
        *   You can also provide direct GCS URIs here if a specific input is not from `input_sources` or a previous task.
    *   `output_prefix`: A string, often a templated GCS path (e.g., `${gcs_paths.output_results_bucket}/${output_prefixes.aligned_bams}`), that defines the base GCS "directory" where output files from this task step will be stored.
    *   `produces_primary_output` (boolean, optional): If `true`, this task is expected to generate a main output file that will be used by a subsequent task configured with `"dynamic_from_previous_task"`.
    *   `primary_output_key` (string, optional): If `produces_primary_output` is `true`, this key specifies which entry in the `output_files` dictionary (returned by `run_computational_task`) should be considered the primary output. For example, if a task produces `{"aligned_sam": "gs://...", "log_file": "gs://..."}` and `primary_output_key` is `"aligned_sam"`, then `"gs://..."` will be passed to the next dynamic task.

## 6. `output_prefixes`

This section defines named relative path components used to organize different types of final outputs within the main `output_results_bucket`.

*   **Role:** Helps structure the output data in GCS.
*   **Key sub-fields:** Each key is a logical name (e.g., `aligned_bams`, `multiqc_reports`), and the value is a relative path string (e.g., `aligned_bams/`).
*   **How it's used:** The `output_prefix` fields in the `workflow_definition` often use these named prefixes to construct full GCS paths for storing task outputs (e.g., `${gcs_paths.output_results_bucket}/${output_prefixes.aligned_bams}`).

## 7. `gcp_costs`

This section specifies the assumed cost rates for Google Cloud Platform (GCP) resources.

*   **Role:** Provides the necessary data for estimating the monetary cost of executing computational tasks.
*   **Key fields:**
    *   `cpu_per_hour`: The cost of one CPU core per hour.
    *   `ram_per_gb_hour`: The cost of one Gigabyte of RAM per hour.
*   **How it's used:** The `run_task.py` script (specifically, the `calculate_gcp_cost` function) uses these rates, along with the allocated resources (CPUs, RAM) and task duration, to estimate the cost of each task. This cost can then be used as part of the reward signal for the RL agent.

## 8. `rl_config`

This section contains parameters that are specifically for configuring the Reinforcement Learning agent's training process and behavior.

*   **Role:** Defines how the RL agent learns, makes decisions, and how its state and rewards are structured.
*   **Key sub-fields (examples):**
    *   `action_tiers`: A dictionary defining the discrete set of resource allocation options (actions) available to the RL agent for each task. Each tier typically specifies a CPU count and RAM amount (e.g., `{0: {cpu: 1, ram_gb: 4}, 1: {cpu: 2, ram_gb: 8}}`).
    *   `reward_weights`: A dictionary specifying the weights for different components of the reward signal (e.g., `success: 10.0, cost: -1.0, time: -0.5, utilization: 0.2`). These weights determine how the agent prioritizes different objectives (e.g., successful completion, minimizing cost/time, maximizing resource utilization).
    *   `state_normalization_constants`: A dictionary containing values used to normalize features in the environment's state representation (e.g., `max_duration_seconds`, `max_cost`). Normalization helps stabilize RL training.
    *   Other parameters like `gamma` (discount factor), `learning_rate`, `fcnet_hiddens` (neural network architecture), `entropy_coeff` (for exploration in PPO), `vf_loss_coeff` (value function loss coefficient for PPO) are typically set here and used by the RLlib algorithm configuration (now PPO) in `controller.py`.
*   **How it's used:** These parameters are primarily consumed by the `WorkflowEnv` class (to define action/observation spaces, calculate rewards, and normalize state) and by the `controller.py` script when setting up the PPO algorithm configuration.

## 9. `gcs_fuse_mounts`

This section lists GCS bucket names that need to be FUSE mounted on the Ray worker nodes.

*   **Role:** Ensures that GCS buckets are accessible as local filesystem paths on worker nodes. This is critical for tools that expect file paths rather than GCS URIs.
*   **Key sub-fields:**
    *   `buckets_to_mount`: An array (list) of GCS bucket names (e.g., `my-genomics-data`, `my-results-bucket`). **Note:** These are just the bucket names, not the full `gs://` paths.
*   **How it's used:**
    *   The `create_ray_cluster_script.py` script reads this list (and dynamically adds buckets from `gcs_paths`) to generate a startup script for Ray cluster nodes.
    *   This startup script typically uses the `gcsfuse_startup.sh.template` to configure and run `gcsfuse` for each specified bucket.
    *   The buckets are mounted under the `gcs_fuse_mount_base` path defined in `gcs_paths`. For example, if `gcs_fuse_mount_base` is `/mnt/gcs/` and `my-bucket` is listed here, tools can access its contents via `/mnt/gcs/my-bucket/`. This allows `run_task.py` to provide local-like paths to the Docker containers.

By understanding and appropriately modifying these sections, you can tailor the system to run diverse genomics workflows, utilize different datasets, and experiment with various RL agent configurations.
