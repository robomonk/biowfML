# How to Run the Genomics Workflow Optimization Code

This document provides instructions on how to run the Reinforcement Learning (RL) agent for optimizing genomics workflows and related components.

## 1. Primary Way to Run: Training the RL Agent

The main entry point for training the RL agent is `controller.py`. This script orchestrates the training process to learn optimal sequences of genomics tasks.

**Command:**

```bash
python controller.py --config path/to/your/config.yaml
```

**Key Command-Line Arguments for `controller.py`:**

*   `--config`: (Required) Path to your `config.yaml` file detailing the genomics pipeline, resources, and GCS paths.
*   `--train-iterations`: Number of training iterations for the RL agent (e.g., `100`).
*   `--num-workers`: Number of Ray worker actors to use for distributed training (e.g., `2`). This should generally match the number of available worker nodes in your Ray cluster minus one for the head node if it also runs workers.
*   `--ray-address`: The address of your Ray cluster.
    *   For a local Ray cluster (started with `ray start --head`): `auto` or `ray://127.0.0.1:10001`
    *   For a remote Ray cluster: `ray://<head_node_ip>:10001`

## 2. Prerequisites for RL Training

Before launching the RL training with `controller.py`, ensure the following prerequisites are met:

*   **Running Ray Cluster:**
    *   You need an active Ray cluster. This can be a local cluster on your machine (for testing) or a multi-node cluster.
    *   Use `ray start --head --port 6379 --ray-client-server-port 10001 --dashboard-port 8265` (adjust ports if needed) on your head node.
    *   Connect worker nodes using the join command provided by `ray start`.
    *   Specify the cluster address to `controller.py` using the `--ray-address` argument.
*   **Valid `config.yaml` File:**
    *   This file is crucial. It defines your genomics workflow, input data, tool configurations, and GCS locations. Pay close attention to:
        *   `gcs_paths`:
            *   `output_results_bucket`: Where final workflow results and metrics will be stored.
            *   `tensorboard_log_bucket`: For storing TensorBoard logs from the RL training.
            *   `staging_bucket`: Used for intermediate data transfer if needed by your setup.
        *   `input_sources`: These must point to valid GCS locations containing your raw genomics data (e.g., FASTQ files, reference genomes).
        *   `tool_images`: Verify that the specified Docker images (e.g., for BWA, Samtools, Picard, GATK) are correct and accessible from your Ray worker nodes.
        *   `gcs_fuse_mounts`:
            *   The system relies on GCS FUSE to mount GCS buckets as local file systems on Ray worker nodes.
            *   Ensure that all GCS buckets that will be accessed for inputs or outputs (derived from `gcs_paths` and `input_sources`) are listed or implicitly covered by the mount configurations.
            *   The `gcsfuse_startup.sh.template` file is used by `create_ray_cluster_script.py` to generate a startup script for Ray nodes, which includes setting up GCS FUSE mounts. If you are setting up your Ray cluster manually, you'll need to ensure GCS FUSE is configured correctly on each worker to mount the necessary buckets under the path specified by `gcs_paths.gcs_fuse_mount_base` in your `config.yaml`.
*   **Docker:**
    *   Docker must be installed and running on all Ray worker nodes. The genomics tasks are executed within Docker containers.
*   **Python Dependencies:**
    *   Install all required Python packages in the environment where you run `controller.py` (and on Ray workers if they don't share the same environment):
        ```bash
        pip install -r requirements.txt
        ```

## 3. Running Unit Tests

Unit tests have been added for some utility functions and the metrics logger. You can run them using Python's built-in unittest discovery:

```bash
python -m unittest discover
```
This command should be run from the root directory of the project.

## 4. Running a Single Genomics Task (Advanced/Informational)

The script `run_task.py` contains the core logic for executing individual genomics tasks (e.g., BWA alignment, Picard sorting) within a Ray actor (`run_computational_task` function). This function is normally invoked by the `WorkflowEnv` during the RL training process.

**Currently, there isn't a dedicated command-line script to run a single, arbitrary genomics task outside of the RL training loop.**

If you need to run a specific task directly (e.g., for testing or debugging a particular step), you would need to write a small Python script that does the following:

1.  **Initialize Ray:**
    ```python
    import ray
    ray.init(address='auto') # Or your specific Ray cluster address
    ```
2.  **Load Pipeline Configuration:**
    ```python
    import yaml
    with open('path/to/your/config.yaml', 'r') as f:
        pipeline_config = yaml.safe_load(f)
    ```
3.  **Import `run_computational_task`:**
    ```python
    from run_task import run_computational_task
    ```
4.  **Define Task Parameters and Call the Remote Function:**
    You'll need to provide all necessary arguments to `run_computational_task.remote(...)`, for example:
    ```python
    task_id = "my_test_bwa_align_01"
    task_type = "bwa_mem_align"
    input_gcs_paths = {
        "ref": "gs://your-bucket/reference/hg38.fasta",
        "read1": "gs://your-bucket/reads/sample1_R1.fastq.gz",
        "read2": "gs://your-bucket/reads/sample1_R2.fastq.gz"
    }
    output_gcs_dir = "gs://your-results-bucket/single_task_outputs/"
    cpu_allocation = 4
    ram_allocation_gb = 8

    # Ensure the run_task.py script (and utils.py) is in your PYTHONPATH
    # or the same directory

    task_future = run_computational_task.remote(
        task_id=task_id,
        task_type=task_type,
        input_gcs_paths=input_gcs_paths,
        output_gcs_dir=output_gcs_dir,
        cpu_allocation=cpu_allocation,
        ram_allocation_gb=ram_allocation_gb,
        pipeline_config=pipeline_config
    )
    result = ray.get(task_future)
    print(f"Task completed. Result: {result}")
    ```
5.  **Prerequisites for `run_computational_task`:**
    *   Ensure all prerequisites for the task itself are met on the Ray worker where it runs. This includes:
        *   GCS FUSE mounts configured correctly for all input and output GCS paths.
        *   Docker installed and able to pull the required `tool_image` from your `config.yaml`.
        *   The `pipeline_config` having correct paths and settings for the task.

This approach is more involved and is generally intended for development or debugging purposes. The primary method for running workflows is through the `controller.py` RL agent.
