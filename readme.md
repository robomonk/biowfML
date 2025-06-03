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

