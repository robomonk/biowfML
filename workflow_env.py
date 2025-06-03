import gym
import numpy as np
import ray
import uuid
import os
import time
from google.cloud import storage
import subprocess
from run_task import run_computational_task # Corrected import to the right function
from metrics_logger import MetricsLogger # Import MetricsLogger for use

class WorkflowEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config: dict):
        """
        Initializes the Workflow Environment.

        Args:
            env_config (dict): A dictionary containing the 'pipeline_config' loaded from config.yaml.
        """
        super(WorkflowEnv, self).__init__()
        self.config = env_config['pipeline_config']

        # --- Extract Configured RL Parameters ---
        self.action_tiers = self.config['rl_config']['action_tiers']
        self.reward_weights = self.config['rl_config']['reward_weights']
        self.normalization_constants = self.config['rl_config']['state_normalization_constants']

        # --- GCS Path Configuration ---
        self.gcs_paths = self.config['gcs_paths']
        self.staging_bucket = self.gcs_paths['staging_bucket']
        self.input_sources = self.config['input_sources']
        self.workflow_definition = self.config['workflow_definition']
        self.output_prefixes = self.config['output_prefixes']

        # --- Define Action and Observation Spaces ---
        # Action Space: Discrete set of resource allocation tiers (0, 1, 2)
        self.action_space = gym.spaces.Discrete(len(self.action_tiers))

        # Observation Space: Needs to match your design (s_task_type, s_input_features, s_attempt, s_hist_perf, s_prev_task_metrics, s_cluster_load)
        # Shape based on designed features:
        # 0: Normalized task type index
        # 1: Normalized input size (for current task)
        # 2: Normalized attempt number (for current task)
        # 3-6: Normalized previous task metrics (duration, cost, cpu_util, ram_util)
        # 7-10: Placeholder for s_hist_perf (avg duration, avg cost, avg cpu, avg ram for this task type)
        # 11: Placeholder for s_cluster_load
        observation_dim = 1 + 1 + 1 + 4 + 4 + 1 # Total 12 features
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(observation_dim,), dtype=np.float32)

        # --- Internal Environment State ---
        self.current_task_idx = 0       # Index of current task in workflow_definition
        self.run_id = None              # Unique ID for current episode/run
        self.staged_gcs_input_paths = {} # GCS paths of inputs after staging (e.g., from public -> staging_bucket)
        self.last_task_output_path = None # Output of previous task, passed as dynamic input to next
        self.metrics_logger_actor = None # MetricsLogger Ray Actor

        # For historical performance (simplified for now)
        self.historical_performance = {} # {task_type: {'durations':[], 'costs':[], ...}}

    def _stage_data(self):
        """
        Copies input data from public/source GCS paths to a unique subdirectory
        within the project's staging bucket. This ensures data is owned and accessible.
        """
        self.run_id = str(uuid.uuid4())
        self.staged_gcs_input_paths = {}
        storage_client = storage.Client()
        staging_bucket_obj = storage_client.bucket(self.staging_bucket)

        print(f"[Env] Staging data for run ID: {self.run_id}")

        for input_key, source_gcs_path_template in self.input_sources.items():
            source_gcs_path = source_gcs_path_template # For input_sources, these are already full GCS paths

            # Extract source bucket name and blob name from the GCS URI
            if not source_gcs_path.startswith("gs://"):
                raise ValueError(f"Input source path must start with 'gs://': {source_gcs_path}")

            source_parts = source_gcs_path.split('//')[1].split('/', 1)
            source_bucket_name = source_parts[0]
            source_blob_name = source_parts[1] if len(source_parts) > 1 else ''

            # Destination path within your staging bucket for this unique run
            destination_blob_name = os.path.join(self.run_id, source_blob_name)

            # Check if destination already exists (for idempotency if run_id is reused)
            destination_blob = staging_bucket_obj.blob(destination_blob_name)
            if destination_blob.exists():
                print(f"[Env] File gs://{self.staging_bucket}/{destination_blob_name} already exists. Skipping copy.")
                self.staged_gcs_input_paths[input_key] = f"gs://{self.staging_bucket}/{destination_blob_name}"
                continue

            # Perform the copy operation
            try:
                source_bucket_obj = storage_client.bucket(source_bucket_name)
                source_blob_obj = source_bucket_obj.blob(source_blob_name)

                # Using rewrite method for efficient server-side copy
                token = source_blob_obj.rewrite(destination_blob)
                while token[0] is not None:
                    token = source_blob_obj.rewrite(destination_blob, token=token[0])

                print(f"[Env] Copied {source_gcs_path} to gs://{self.staging_bucket}/{destination_blob_name}")
                self.staged_gcs_input_paths[input_key] = f"gs://{self.staging_bucket}/{destination_blob_name}"
            except Exception as e:
                print(f"[Env] Error copying {source_gcs_path} to staging: {e}")
                self.staged_gcs_input_paths[input_key] = None # Mark as failed
                raise # Re-raise to stop episode if staging fails critically

    def reset(self):
        """
        Resets the environment for a new episode.
        Stages data and prepares the first task.
        """
        print("\n[Env] Resetting environment for a new episode...")
        self.current_task_idx = 0
        self.last_task_output_path = None # Reset output from previous episode
        self.staged_gcs_input_paths = {} # Reset staged paths

        # Initialize metrics logger actor once per environment, if not already done
        if self.metrics_logger_actor is None:
             metrics_gcs_log_dir = self.gcs_paths['output_results_bucket'] + "/task_metrics/" # Path for task-level logs
             self.metrics_logger_actor = MetricsLogger.remote(metrics_gcs_log_dir) # Initialize Ray Actor

        # Stage data from public sources to our staging bucket for this run
        try:
            self._stage_data()
        except Exception as e:
            print(f"[Env] FATAL ERROR during data staging: {e}. Cannot proceed with episode.")
            # Return an observation that indicates failure and potentially end episode (done=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32) # Indicate failure state
            # Or raise an exception if you want to stop the training.

        # Get the first task definition in the workflow
        first_task_def = self.workflow_definition[self.current_task_idx]

        # Initial observation with default/empty previous task results
        initial_observation = self._calculate_observation(
            task_result={}, # Empty for first observation
            current_task_def=first_task_def,
            attempt_num=0,
            run_id=self.run_id
        )
        print(f"[Env] Reset complete. Initial observation: {initial_observation}")
        return initial_observation

    def step(self, action):
        """
        Executes one step in the environment (runs one computational task).
        Manages the sequential execution of tasks from workflow_definition.
        """
        # Check if episode is already done (e.g., if previous task failed)
        if self.current_task_idx >= len(self.workflow_definition):
            # If done, subsequent steps should return done=True with a final state
            # This state indicates the end of the episode.
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, {"message": "Episode already completed."}


        current_task_def = self.workflow_definition[self.current_task_idx]
        task_type = current_task_def['type']
        task_id = f"{current_task_def['task_id_prefix']}-{self.run_id}-{self.current_task_idx}" # Unique ID for this task
        attempt_num = 0 # Currently, retries are not handled in env, always attempt 0

        cpu_allocation = self.action_tiers[action]['cpu']
        ram_allocation_gb = self.action_tiers[action]['ram_gb']

        print(f"\n[Env] Step {self.current_task_idx+1}/{len(self.workflow_definition)}: Task '{task_type}' (ID: {task_id}) with action {action} (CPU={cpu_allocation}, RAM={ram_allocation}GB)")

        # --- Resolve Input Paths for the current task ---
        # Inputs can be from staged data or output of previous tasks.
        task_input_gcs_paths = {}
        for input_key, input_val_template in current_task_def['inputs'].items():
            if isinstance(input_val_template, str):
                if input_val_template.startswith("${input_sources."):
                    # This refers to a public input that has been staged
                    # e.g., "${input_sources.ref_genome_chr19}"
                    source_key = input_val_template.split('.')[1].rstrip('}')
                    resolved_path = self.staged_gcs_input_paths.get(source_key)
                    if not resolved_path:
                        raise ValueError(f"Staged path for '{source_key}' not found for task '{task_type}'. Staging failed or key missing in config.")
                    task_input_gcs_paths[input_key] = resolved_path
                elif input_val_template == "dynamic_from_previous_task":
                    # This input comes from the primary output of the immediately preceding task
                    if self.last_task_output_path is None:
                        raise RuntimeError(f"Task '{task_type}' expects 'dynamic_from_previous_task' but previous task did not produce a primary output or failed.")
                    task_input_gcs_paths[input_key] = self.last_task_output_path
                elif input_val_template == "dynamic_from_previous_task_intervals":
                    # Special case for GATK IndelRealigner needing intervals.
                    # This assumes last_task_output_path holds the intervals file from RealignerTargetCreator.
                    if self.last_task_output_path is None:
                         raise RuntimeError(f"Task '{task_type}' expects 'dynamic_from_previous_task_intervals' but previous task did not produce the intervals file.")
                    task_input_gcs_paths[input_key] = self.last_task_output_path
                else:
                    # Direct GCS path if not a template or dynamic input
                    task_input_gcs_paths[input_key] = input_val_template
            else: # For non-string inputs (e.g., if you had lists or dicts as inputs in config)
                task_input_gcs_paths[input_key] = input_val_template


        # Resolve output GCS directory for the current task
        # The output_prefix in config.yaml is already a full GCS path
        task_output_gcs_dir = current_task_def['output_prefix']

        # --- Launch the task asynchronously on Ray ---
        task_result = {} # Initialize task_result in case of pre-execution error
        try:
            task_ref = run_computational_task.remote(
                task_id,
                task_type,
                task_input_gcs_paths,
                task_output_gcs_dir,
                cpu_allocation,
                ram_allocation_gb,
                self.config # Pass the full pipeline config to run_task
            )
            task_result = ray.get(task_ref) # Wait for the task to complete
            print(f"[Env] Task {task_id} completed. Success: {task_result['success']}")
        except Exception as e:
            print(f"[Env] Error running task {task_id}: {e}")
            # Create a failed task result to allow reward calculation and episode continuation/termination
            task_result = {
                "task_id": task_id, "task_type": task_type, "duration_seconds": -1,
                "cost": self.normalization_constants['max_cost'], # Penalize with max cost
                "cpu_utilization": 0, "ram_utilization_gb": 0,
                "cpu_allocated": cpu_allocation, "ram_allocated_gb": ram_allocation_gb,
                "exit_status": 1, "success": False, "output_files": {}
            }

        # Log task metrics to GCS using the MetricsLogger Actor
        if self.metrics_logger_actor:
            ray.get(self.metrics_logger_actor.log_task_metrics.remote(task_result)) # Log asynchronously

        # --- Update Environment State for Next Step ---
        self.last_task_result = task_result
        self.last_task_output_path = None # Reset for the next task
        if task_result['success'] and current_task_def.get('produces_primary_output', False):
            # The primary output is what the next task might consume.
            # Assuming the first value in output_files is the primary one.
            if task_result['output_files']:
                self.last_task_output_path = next(iter(task_result['output_files'].values()))
            print(f"[Env] Primary output from '{task_type}' (ID: {task_id}): {self.last_task_output_path}")

        # --- Calculate Reward ---
        reward = self._calculate_reward(task_result)

        # --- Determine Episode Progression ---
        self.current_task_idx += 1
        # Episode is done if all tasks are completed OR if the current task failed critically
        done = self.current_task_idx >= len(self.workflow_definition) or not task_result['success']

        next_observation = self._calculate_observation(
            task_result=task_result,
            current_task_def=self.workflow_definition[self.current_task_idx] if not done else None, # Pass None if episode is done
            attempt_num=attempt_num,
            run_id=self.run_id
        )

        info = {
            "task_id": task_id,
            "task_type": task_type,
            "cpu_allocation": cpu_allocation,
            "ram_allocation": ram_allocation,
            "result": task_result,
            "run_id": self.run_id # Include run_id in info for debugging
        }

        return next_observation, reward, done, info

    def _calculate_reward(self, task_result: dict) -> float:
        """
        Calculates the composite reward based on task results and configured weights.
        Maximizing this composite value aims to minimize cost/time and maximize utilization/success.
        """
        success = task_result.get('success', False)
        duration = task_result.get('duration_seconds', self.normalization_constants['max_duration_seconds'])
        cost = task_result.get('cost', self.normalization_constants['max_cost'])
        cpu_util_ratio = task_result.get('cpu_utilization', 0.0) # Already normalized 0-1 ratio from run_task
        ram_util_gb = task_result.get('ram_utilization_gb', 0.0) # Actual GB used

        # Rsuccess: binary (+1 for successful completion, -1 for failure)
        r_success = self.reward_weights['success'] * (1 if success else -1)

        # Rcost: normalized negative cost (higher cost -> lower reward)
        # Clip cost to max_cost for normalization to prevent extreme values.
        norm_cost = min(cost / self.normalization_constants['max_cost'], 1.0)
        r_cost = self.reward_weights['cost'] * norm_cost

        # Rtime: normalized negative execution time (longer time -> lower reward)
        # Clip duration to max_duration for normalization.
        norm_time = min(duration / self.normalization_constants['max_duration_seconds'], 1.0)
        r_time = self.reward_weights['time'] * norm_time

        # Rutil: (simplified for now) - Higher utilization is better.
        # Convert RAM utilization to a ratio based on allocated RAM for the utility component.
        ram_allocated_gb = task_result.get('ram_allocated_gb', 1) # Prevent division by zero
        ram_util_ratio = ram_util_gb / ram_allocated_gb if ram_allocated_gb > 0 else 0.0

        # Average CPU and RAM utilization ratios.
        avg_util_ratio = (cpu_util_ratio + ram_util_ratio) / 2.0
        r_util = self.reward_weights['utilization'] * avg_util_ratio # Positive weight for higher average utilization

        total_reward = r_success + r_cost + r_time + r_util

        print(f"[Env] Reward components - Success: {r_success:.2f}, Cost: {r_cost:.2f}, Time: {r_time:.2f}, Util: {r_util:.2f} -> Total Reward: {total_reward:.2f}")

        return float(total_reward)

    def _calculate_observation(self, task_result: dict, current_task_def: dict, attempt_num: int, run_id: str) -> np.ndarray:
        """
        Calculates the observation vector for the DRL agent based on the design document.
        """
        obs_vec = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Feature 0: Normalized Task Type Index (Categorical/numerical representation of task type)
        if current_task_def:
            # Find the index of the current task type within the workflow definition
            # This provides a unique normalized ID for each distinct task step.
            # E.g., bwa_index=0, samtools_faidx=1, etc.
            task_type_names = [t_def['type'] for t_def in self.workflow_definition]
            try:
                task_type_idx = task_type_names.index(current_task_def['type'])
                obs_vec[0] = task_type_idx / (len(task_type_names) - 1 if len(task_type_names) > 1 else 1.0) # Normalize 0 to 1
            except ValueError:
                obs_vec[0] = 0.0 # Should not happen if task_type is from workflow_definition
        else: # If episode is done, current_task_def is None
            obs_vec[0] = 0.0 # Or some specific value to indicate end of tasks

        # Feature 1: Normalized Input Size (total size/count of input files for the *current* task)
        # This is hard to calculate dynamically without inspecting GCS files for each task.
        # For simplicity, let's derive it from the total staged input size.
        # In a more advanced setup, you'd calculate actual input size for the specific task.
        total_staged_input_size_gb = 0.0
        # You would need to add logic here to compute file sizes of self.staged_gcs_input_paths
        # For now, let's use a dummy value or a rough estimate.
        # The most accurate way is to get blob.size for each input file.
        obs_vec[1] = 0.5 # Placeholder for input size, normalized to 0-1

        # Feature 2: Normalized Attempt Number
        obs_vec[2] = min(attempt_num / self.normalization_constants.get('max_attempts', 5.0), 1.0) # Max 5 attempts for normalization

        # Features 3-6: Normalized s_prev_task_metrics (duration, cost, cpu_util, ram_util)
        prev_duration = task_result.get('duration_seconds', 0)
        prev_cost = task_result.get('cost', 0)
        prev_cpu_util = task_result.get('cpu_utilization', 0.0) # Already 0-1 ratio
        prev_ram_util_gb = task_result.get('ram_utilization_gb', 0.0) # Actual GB

        obs_vec[3] = min(prev_duration / self.normalization_constants['max_duration_seconds'], 1.0)
        obs_vec[4] = min(prev_cost / self.normalization_constants['max_cost'], 1.0)
        obs_vec[5] = prev_cpu_util # Assumed to be 0-1 ratio from run_task
        # Normalize RAM utilization by its allocated amount for the previous task
        prev_ram_allocated_gb = task_result.get('ram_allocated_gb', 1)
        obs_vec[6] = min(prev_ram_util_gb / prev_ram_allocated_gb if prev_ram_allocated_gb > 0 else 0.0, 1.0)

        # Features 7-10: Placeholder for s_hist_perf (avg duration, avg cost, avg cpu, avg ram for this task type)
        # This would require storing and updating historical metrics over episodes/tasks.
        # For now, using zeros or fixed small values.
        # In a real system, you'd query historical_performance or a database.
        obs_vec[7:11] = 0.0 # Placeholder for historical performance

        # Feature 11: Placeholder for s_cluster_load (e.g., # of active/pending tasks, available resources)
        # This would require querying Ray's internal API or K8s API.
        # For now, using zero or fixed small value.
        obs_vec[11] = 0.0 # Placeholder for cluster load

        return obs_vec


    def render(self, mode='human'):
        # Optional: Print current state or task info
        if hasattr(self, 'current_task_idx') and self.current_task_idx > 0 and self.last_task_result:
            print(f"Env State - Current Task Index: {self.current_task_idx}, Last Reward: {self._calculate_reward(self.last_task_result):.2f}")
        else:
            print("Env State - Initializing/Waiting for first task...")


    def close(self):
        """
        Clean up resources (e.g., Ray Actors if they are managed by the environment).
        """
        # No specific Ray shutdown needed here, as controller manages Ray.
        # If MetricsLogger was created here, it might be stopped here.
        pass