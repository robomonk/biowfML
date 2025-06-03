import ray
from ray.rllib.algorithms.ppo import PPOConfig # Changed A3C to PPO
from ray.air.config import RunConfig # Added RunConfig
from ray.tune.registry import register_env
import os
import yaml
import argparse # For command-line arguments
import logging # Added logging

# Import your custom environment and other components
from workflow_env import WorkflowEnv
# from metrics_logger import MetricsLogger # MetricsLogger is initialized by WorkflowEnv

# --- Command-line Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run DRL-optimized genomics pipeline training.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file (e.g., config.yaml)')
    parser.add_argument('--train-iterations', type=int, default=100,
                        help='Number of training iterations.')
    parser.add_argument('--num-workers', type=int, default=1, # Start with 1 for local testing, 4+ for cluster
                        help='Number of parallel rollout workers for RLlib.')
    parser.add_argument('--ray-address', type=str, default='auto',
                        help='Address of the Ray cluster (e.g., "auto" or "vertex_ray://<cluster_name>")')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --- Basic Logging Configuration ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] # Ensure logs go to console
    )
    logger = logging.getLogger(__name__)

    # --- 1. Load Configuration ---
    if not os.path.exists(args.config):
        # No logger configured yet if this fails early, so using print/raise
        print(f"FATAL: Config file not found at: {args.config}")
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    with open(args.config, 'r') as f:
        PIPELINE_CONFIG = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {args.config}")

    # --- Extract RLlib-specific config ---
    rl_config_params = PIPELINE_CONFIG['rl_config']
    gcs_paths = PIPELINE_CONFIG['gcs_paths']

    # Determine GCS paths for TensorBoard logs and Checkpoints
    # Ensure they are full GCS URIs
    base_gcs_bucket = gcs_paths['tensorboard_log_bucket']
    if not base_gcs_bucket.startswith("gs://"):
        # Assuming it's just a bucket name, prepend gs://, otherwise, this logic might need adjustment
        # if tensorboard_log_bucket could be a sub-path itself. For now, assume it's a bucket name.
        # However, the config.yaml implies it's just the bucket name.
        base_gcs_bucket = f"gs://{base_gcs_bucket}"

    TENSORBOARD_LOG_DIR = f"{base_gcs_bucket.rstrip('/')}/tensorboard_logs/"
    CHECKPOINT_DIR = f"{base_gcs_bucket.rstrip('/')}/checkpoints/"

    # --- 2. Initialize Ray ---
    # Connect to the existing Ray cluster
    try:
        # Pass the current directory to workers, so they can find workflow_env and utils
        # This assumes controller.py, workflow_env.py, and utils.py are in the same directory.
        # For a more robust solution with complex project structures, consider packaging.
        import sys
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Set working_dir for workers to the directory containing the scripts.
        # This helps ensure that workflow_env.py and utils.py are importable.
        runtime_env = {"working_dir": current_script_dir}

        ray.init(address=args.ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
        # logger.info(f"Ray initialized with address: {ray.get_head_address()}") # Commented out, problematic with Ray Client
    except Exception as e:
        logger.exception("Error initializing Ray:")
        exit(1)

    # --- 3. Register Workflow Environment ---
    # Register your custom gym environment with RLlib's registry
    register_env("WorkflowEnv", lambda env_config: WorkflowEnv(env_config))
    logger.info("WorkflowEnv registered with RLlib.")

    # --- 4. Configure RLlib PPO Algorithm --- # Changed A3C to PPO
    # First, create the PPOConfig object with all chained settings
    ppo_config_object = (
        PPOConfig() # Changed A3CConfig to PPOConfig
        .environment(env="WorkflowEnv", env_config={'pipeline_config': PIPELINE_CONFIG}) # Pass full config to env
        .env_runners(num_env_runners=args.num_workers) # Changed from .rollouts()
        .framework("torch") # Use PyTorch for the neural network
        .training(
            gamma=rl_config_params.get('gamma', 0.99), # Discount factor
            lr=rl_config_params.get('learning_rate', 0.0001), # Learning rate
            model={
                "fcnet_hiddens": rl_config_params.get('fcnet_hiddens', [256, 256]) # NN hidden layers
            },
            entropy_coeff=rl_config_params.get('entropy_coeff', 0.01), # Encourage exploration
            vf_loss_coeff=rl_config_params.get('vf_loss_coeff', 0.5), # Value function loss coefficient
            # n_step_max=rl_config_params.get('n_step_max', 50), # N-step returns (Removed for PPO)
        )
        # RunConfig parameters like storage_path and name are typically handled by Tuner or higher-level Trainable.
        # For direct Algo usage, results often go to a default local dir (e.g. ~/ray_results/exp_name)
        # We will rely on algo.save(checkpoint_dir=CHECKPOINT_DIR) for explicit checkpointing.
        # Tensorboard logging location might need to be inferred from algo.logdir or configured differently.
        .resources(num_gpus=rl_config_params.get('num_gpus', 0)) # Set 0 if no GPUs, >0 if GPUs for NN training
        .debugging(log_level="INFO") # Set log level
        # Removed .build() from the end of the config chain
    )

    # Ensure correct observation and action spaces are set (for debugging)
    # print(f"Observation space: {ppo_config_object.observation_space}") # Would need to build env first or get from built algo
    # print(f"Action space: {ppo_config_object.action_space}")

    # --- 5. Build and Train the Algorithm ---
    # Build the algorithm instance from the config object
    algo = ppo_config_object.build_algo()

    # Setup TensorBoard logging to GCS
    # RLlib's Logger.log_dir is the base for TensorBoard event files
    # We'll set the result_dir (where Tune saves experiment results) to GCS
    from ray.tune.logger import pretty_print
    # Removed: from ray.tune.experiment.trial import Trial (unused)

    # Removed tune_config dictionary, as RunConfig in .training() handles log location
    # and checkpointing is manual via algo.save().

    logger.info(f"TensorBoard logs will be saved to GCS path specified in RunConfig: {TENSORBOARD_LOG_DIR}")
    logger.info(f"Manual checkpoints will be saved to: {CHECKPOINT_DIR}")


    # Training loop
    logger.info(f"Starting DRL training for {args.train_iterations} iterations...")
    for i in range(args.train_iterations):
        result = algo.train()

        # Print key metrics from the training result
        logger.info(f"Iteration: {i+1}")
        # Ensure pretty_print(result) is properly formatted, potentially multi-line
        logger.info(f"Training result for iteration {i+1}:\n{pretty_print(result)}")


        # RLlib automatically handles checkpointing to local_dir/upload_dir based on config.
        # We can also manually save more explicitly if needed.
        if (i + 1) % rl_config_params.get('checkpoint_freq', 10) == 0 or (i + 1) == args.train_iterations:
            # RLlib saves checkpoints to a subdirectory within its logging path.
            # The returned path is usually a local path on the head node.
            # For GCS, it gets synced by Tune's upload_dir.
            current_checkpoint_path = algo.save(checkpoint_dir=CHECKPOINT_DIR)
            logger.info(f"Checkpoint saved in: {current_checkpoint_path}")

    logger.info("DRL training complete.")

    # --- 6. Shut down Ray ---
    # Only shutdown if this script is responsible for managing the whole Ray lifecycle
    # If the Ray cluster is persistent, you might not want to shut it down here.
    # ray.shutdown() # Uncomment if you want to shutdown Ray client connection