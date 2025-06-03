import gymnasium as gym # Changed from import gym
import numpy as np
import ray
import uuid
import os
import time
from google.cloud import storage
# import subprocess # Removed
import statistics # Added for calculating mean
import logging # Added logging
from run_task import run_computational_task # Corrected import to the right function
from metrics_logger import MetricsLogger # Import MetricsLogger for use
from utils import parse_gcs_uri, InvalidGCSPathError # Added

logger = logging.getLogger(__name__)

class StagingErrorException(Exception):
    pass

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
        self.gcs_client = None # Initialize GCS client, will be instantiated on first use

        # For historical performance
        self.historical_performance = {} # {task_type: {'durations':[], 'costs':[], 'cpu_utils':[], 'ram_utils':[]}}

    def _resolve_input_gcs_paths_for_observation(self, current_task_def: dict) -> list[str]:
        if not current_task_def:
            return []

        task_input_gcs_uris = []
        for input_key, input_val_template in current_task_def.get('inputs', {}).items():
            resolved_path = None
            if isinstance(input_val_template, str):
                if input_val_template.startswith("${input_sources."):
                    source_key = input_val_template.split('.')[1].rstrip('}')
                    resolved_path = self.staged_gcs_input_paths.get(source_key)
                elif input_val_template == "dynamic_from_previous_task":
                    resolved_path = self.last_task_output_path # This comes from the main step logic
                elif input_val_template == "dynamic_from_previous_task_intervals":
                     resolved_path = self.last_task_output_path # Assuming this is how it's handled
                else: # Direct GCS path
                    if input_val_template.startswith("gs://"):
                        resolved_path = input_val_template

            if resolved_path and resolved_path.startswith("gs://"):
                task_input_gcs_uris.append(resolved_path)
            elif resolved_path: # Could be a local path if workflow changes, log a warning
                logger.warning(f"Resolved input path {resolved_path} is not a GCS URI for size calculation.")

        return task_input_gcs_uris

    def _stage_data(self):
        """
        Copies input data from public/source GCS paths to a unique subdirectory
        within the project's staging bucket. This ensures data is owned and accessible.
        """
        self.run_id = str(uuid.uuid4())
        self.staged_gcs_input_paths = {}
        storage_client = storage.Client()
        staging_bucket_obj = storage_client.bucket(self.staging_bucket)

        logger.info(f"Staging data for run ID: {self.run_id}")

        for input_key, source_gcs_path_template in self.input_sources.items():
            source_gcs_path = source_gcs_path_template # For input_sources, these are already full GCS paths

            # Extract source bucket name and blob name from the GCS URI
            if not source_gcs_path.startswith("gs://"): # This check is good, but parse_gcs_uri also validates
                logger.error(f"Invalid source GCS path in input_sources (must start with gs://): {source_gcs_path}")
                self.staged_gcs_input_paths[input_key] = None
                raise StagingErrorException(f"Invalid GCS path for staging: {source_gcs_path}")

            try:
                source_bucket_name, source_blob_name = parse_gcs_uri(source_gcs_path)
            except InvalidGCSPathError as e:
                logger.error(f"Invalid source GCS path in input_sources: {source_gcs_path} - {e}")
                self.staged_gcs_input_paths[input_key] = None
                raise StagingErrorException(f"Invalid GCS path for staging: {source_gcs_path} - {e}") from e

            # Destination path within your staging bucket for this unique run
            destination_blob_name = os.path.join(self.run_id, source_blob_name)

            # Check if destination already exists (for idempotency if run_id is reused)
            destination_blob = staging_bucket_obj.blob(destination_blob_name)
            if destination_blob.exists():
                logger.info(f"File gs://{self.staging_bucket}/{destination_blob_name} already exists. Skipping copy.")
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

                logger.info(f"Copied {source_gcs_path} to gs://{self.staging_bucket}/{destination_blob_name}")
                self.staged_gcs_input_paths[input_key] = f"gs://{self.staging_bucket}/{destination_blob_name}"
            except Exception as e:
                logger.exception(f"Error copying {source_gcs_path} to staging for run_id {self.run_id}:")
                self.staged_gcs_input_paths[input_key] = None # Mark as failed
                raise # Re-raise to stop episode if staging fails critically

    def reset(self, *, seed=None, options=None): # Added seed and options
        """
        Resets the environment for a new episode.
        Stages data and prepares the first task.
        """
        logger.info("Resetting environment for a new episode...")
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
        except StagingErrorException: # Already logged in _stage_data or if it's a different exception
            logger.exception(f"A StagingErrorException occurred during reset for run_id {self.run_id}. Cannot proceed.")
            raise # Re-raise to be handled by the agent/training loop
        except Exception as e: # Catch any other unexpected errors during staging setup
            logger.exception(f"Unexpected FATAL ERROR during data staging setup for run_id {self.run_id}: {e}")
            # Convert to StagingErrorException or a more generic EnvSetupException if needed
            raise StagingErrorException(f"Unexpected data staging setup failed for run_id {self.run_id}: {e}") from e

        # Get the first task definition in the workflow
        first_task_def = self.workflow_definition[self.current_task_idx]

        # Initial observation with default/empty previous task results
        initial_observation = self._calculate_observation(
            task_result={}, # Empty for first observation
            current_task_def=first_task_def,
            attempt_num=0,
            run_id=self.run_id
        )
        logger.info(f"Reset complete. Initial observation: {initial_observation}")
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

        logger.info(f"Step {self.current_task_idx+1}/{len(self.workflow_definition)}: Task '{task_type}' (ID: {task_id}) with action {action} (CPU={cpu_allocation}, RAM={ram_allocation_gb}GB)")

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
            logger.info(f"Task {task_id} completed. Success: {task_result['success']}")
        except Exception as e:
            logger.exception(f"Error running task {task_id} via Ray:")
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

        # --- Update Historical Performance ---
        if task_result and task_result.get('success'): # Only log successful tasks for historical averages
            task_type_hist_key = task_result['task_type'] # Use the actual task_type string as key
            if task_type_hist_key not in self.historical_performance:
                self.historical_performance[task_type_hist_key] = {'durations': [], 'costs': [], 'cpu_utils': [], 'ram_utils': []}

            self.historical_performance[task_type_hist_key]['durations'].append(task_result['duration_seconds'])
            self.historical_performance[task_type_hist_key]['costs'].append(task_result['cost'])
            self.historical_performance[task_type_hist_key]['cpu_utils'].append(task_result['cpu_utilization']) # This should be 0-1 ratio

            allocated_ram_gb = task_result.get('ram_allocated_gb', 1)
            actual_ram_gb = task_result.get('ram_utilization_gb', 0)
            ram_util_ratio = actual_ram_gb / allocated_ram_gb if allocated_ram_gb > 0 else 0.0
            self.historical_performance[task_type_hist_key]['ram_utils'].append(ram_util_ratio)

            history_limit = self.config.get('rl_config', {}).get('historical_performance_window', 10)
            for key in self.historical_performance[task_type_hist_key]:
                self.historical_performance[task_type_hist_key][key] = self.historical_performance[task_type_hist_key][key][-history_limit:]

        # --- Update Environment State for Next Step ---
        self.last_task_result = task_result
        self.last_task_output_path = None # Reset for the next task
        if task_result['success'] and current_task_def.get('produces_primary_output', False):
            primary_key = current_task_def.get('primary_output_key')
            task_outputs = task_result.get('output_files', {})
            if primary_key and primary_key in task_outputs:
                self.last_task_output_path = task_outputs[primary_key]
                logger.info(f"Primary output from '{task_type}' (ID: {task_id}) using key '{primary_key}': {self.last_task_output_path}")
            elif primary_key: # Key was specified but not found in outputs
                logger.warning(f"Task {task_id} ('{task_type}') was marked 'produces_primary_output' with key '{primary_key}', but key not found in task_result['output_files'] ({list(task_outputs.keys())}). No primary output set.")
                self.last_task_output_path = None # Explicitly set to None
            else: # produces_primary_output was true, but no key specified in config
                logger.warning(f"Task {task_id} ('{task_type}') is marked 'produces_primary_output' but 'primary_output_key' is missing in its definition in config.yaml. Cannot determine primary output. No primary output set.")
                self.last_task_output_path = None # Explicitly set to None

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

        logger.info(f"Reward components for task {task_id} - Success: {r_success:.2f}, Cost: {r_cost:.2f}, Time: {r_time:.2f}, Util: {r_util:.2f} -> Total Reward: {total_reward:.2f}")

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

        # Feature 1: Normalized Input Size
        if self.gcs_client is None:
            self.gcs_client = storage.Client()

        current_task_input_gcs_uris = self._resolve_input_gcs_paths_for_observation(current_task_def)
        total_input_bytes = 0
        if current_task_def and current_task_input_gcs_uris:
            for gcs_uri in current_task_input_gcs_uris:
                try:
                    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
                    if blob_name: # Ensure blob_name is not empty (e.g. for bucket root, though unlikely for files)
                        blob = self.gcs_client.bucket(bucket_name).get_blob(blob_name)
                        if blob and blob.size is not None:
                            total_input_bytes += blob.size
                        else:
                            logger.warning(f"Could not get blob or size for GCS URI {gcs_uri} when calculating observation.")
                    else: # blob_name is empty, means it's likely just "gs://bucket/"
                        logger.warning(f"Skipping GCS URI with empty blob name for size calculation: {gcs_uri}")
                        continue # Skip to next URI
                except InvalidGCSPathError as e:
                    logger.warning(f"Skipping invalid GCS URI for size calculation: {gcs_uri} - {e}")
                    continue # Skip this URI
                except Exception as e: # Catch other potential errors from GCS client
                    logger.error(f"Error accessing GCS URI {gcs_uri} for size calculation: {e}")
                    continue # Skip this URI

        total_input_gb = total_input_bytes / (1024**3)
        max_input_gb = self.normalization_constants.get('max_input_size_gb', 100.0)
        obs_vec[1] = min(total_input_gb / max_input_gb, 1.0)

        # Feature 2: Normalized Attempt Number
        max_attempts_val = self.normalization_constants.get('max_attempts', 5.0) # Default if somehow missing
        obs_vec[2] = min(attempt_num / max_attempts_val, 1.0)

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

        # Features 7-10: Historical Performance
        avg_hist_duration = 0.0
        avg_hist_cost = 0.0
        avg_hist_cpu_util = 0.0
        avg_hist_ram_util = 0.0

        if current_task_def:
            task_type_history = self.historical_performance.get(current_task_def['type'])
            if task_type_history:
                if task_type_history['durations']:
                    avg_hist_duration = statistics.mean(task_type_history['durations'])
                if task_type_history['costs']:
                    avg_hist_cost = statistics.mean(task_type_history['costs'])
                if task_type_history['cpu_utils']:
                    avg_hist_cpu_util = statistics.mean(task_type_history['cpu_utils'])
                if task_type_history['ram_utils']:
                    avg_hist_ram_util = statistics.mean(task_type_history['ram_utils'])

        obs_vec[7] = min(avg_hist_duration / self.normalization_constants['max_duration_seconds'], 1.0)
        obs_vec[8] = min(avg_hist_cost / self.normalization_constants['max_cost'], 1.0)
        obs_vec[9] = avg_hist_cpu_util  # Assumes CPU util is already 0-1
        obs_vec[10] = avg_hist_ram_util # Assumes RAM util here is also a 0-1 ratio

        # Feature 11: Cluster Load
        try:
            cluster_resources = ray.cluster_resources()
            total_cpus = cluster_resources.get('CPU', 0.0)
            available_cpus = ray.available_resources().get('CPU', 0.0)

            if total_cpus > 0:
                cluster_cpu_load = (total_cpus - available_cpus) / total_cpus
            else:
                cluster_cpu_load = 0.0

            max_load_norm = self.normalization_constants.get('max_cluster_load_percent', 1.0)
            obs_vec[11] = min(cluster_cpu_load / max_load_norm, 1.0)

        except Exception as e:
            logger.error(f"Error getting Ray cluster resources for observation: {e}")
            obs_vec[11] = 0.0

        return obs_vec


    def render(self, mode='human'):
        # Optional: Print current state or task info
        if hasattr(self, 'last_task_result') and self.last_task_result: # Check if last_task_result exists
             current_task_desc = f"Task Index: {self.current_task_idx}"
             if self.current_task_idx < len(self.workflow_definition):
                 current_task_desc = f"Next Task: {self.workflow_definition[self.current_task_idx]['type']}"
             else:
                 current_task_desc = "Workflow Complete"

             logger.info(f"Render Env State - {current_task_desc}, Last Reward: {self._calculate_reward(self.last_task_result):.2f}")
        else:
             logger.info("Render Env State - Initializing or no task processed yet.")


    def close(self):
        """
        Clean up resources (e.g., Ray Actors if they are managed by the environment).
        """
        # No specific Ray shutdown needed here, as controller manages Ray.
        # If MetricsLogger was created here, it might be stopped here.
        pass