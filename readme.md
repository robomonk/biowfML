RL-Optimized Workflow Execution for FASTQ Processing: Full Design Specification

Objective

Design and implement a reinforcement learning (RL)–driven system for dynamically optimizing cloud-based execution of FASTQ data processing pipelines. The goal is to minimize compute cost and latency by learning optimal CPU/RAM allocations and concurrency levels per task using the A3C algorithm, while handling asynchronous execution and partial observability in a cloud-native environment.

Functional Requirements

Support real-world bioinformatics tools (e.g., bwa, samtools).

Execute on Google Cloud using GKE or Vertex AI infrastructure.

Launch and monitor Ray clusters via Helm or Terraform.

Store inputs/outputs and logs in Google Cloud Storage.

Use reinforcement learning (via Ray RLlib) to adaptively control resource allocations per task.

Include cost modeling based on real GCP pricing for CPU/RAM usage.

Enable concurrent execution of multiple tasks.

Include full observability, task logging, and optional web-based monitoring.

## Concurrency and Algorithm Selection

Multiple workflow channels can be processed at once by launching several RLlib environment workers (`--num-workers` in `controller.py`). Each worker runs the pipeline independently so Ray executes tasks in parallel as soon as their inputs become available. The environment maintains an internal queue of pending Ray tasks and polls for completed jobs without blocking. As soon as a worker finishes, the next task is submitted, keeping the pipeline saturated up to the limit defined by `rl_config.max_concurrent_tasks` in `config.yaml`.

The reinforcement learning algorithm is also configurable. Set `rl_config.algorithm` to `ppo`, `a3c`, or `dqn` to choose between Proximal Policy Optimization, Asynchronous Advantage Actor-Critic, or Deep Q-Network training respectively.

