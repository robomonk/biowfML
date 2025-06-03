import ray
from ray.rllib.algorithms.a3c import A3CConfig
from ray.tune.registry import register_env
import os
import yaml
import argparse # For command-line arguments

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

    # --- 1. Load Configuration ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    with open(args.config, 'r') as f:
        PIPELINE_CONFIG = yaml.safe_load(f)

    print(f"Loaded configuration from {args.config}")

    # --- Extract RLlib-specific config ---
    rl_config_params = PIPELINE_CONFIG['rl_config']
    gcs_paths = PIPELINE_CONFIG['gcs_paths']

    # Determine GCS paths for TensorBoard logs and Checkpoints
    TENSORBOARD_LOG_DIR = gcs_paths['tensorboard_log_bucket'] + "/tensorboard_logs/"
    CHECKPOINT_DIR = gcs_paths['tensorboard_log_bucket'] + "/checkpoints/" # Using same bucket for simplicity

    # --- 2. Initialize Ray ---
    # Connect to the existing Ray cluster
    try:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        print(f"Ray initialized with address: {ray.get_head_address()}")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        exit(1)

    # --- 3. Register Workflow Environment ---
    # Register your custom gym environment with RLlib's registry
    register_env("WorkflowEnv", lambda env_config: WorkflowEnv(env_config))
    print("WorkflowEnv registered with RLlib.")

    # --- 4. Configure RLlib A3C Algorithm ---
    config = (
        A3CConfig()
        .environment(env="WorkflowEnv", env_config={'pipeline_config': PIPELINE_CONFIG}) # Pass full config to env
        .rollouts(num_rollout_workers=args.num_workers) # Number of parallel environments
        .framework("torch") # Use PyTorch for the neural network
        .training(
            gamma=rl_config_params.get('gamma', 0.99), # Discount factor
            lr=rl_config_params.get('learning_rate', 0.0001), # Learning rate
            model={
                "fcnet_hiddens": rl_config_params.get('fcnet_hiddens', [256, 256]) # NN hidden layers
            },
            entropy_coeff=rl_config_params.get('entropy_coeff', 0.01), # Encourage exploration
            vf_loss_coeff=rl_config_params.get('vf_loss_coeff', 0.5), # Value function loss coefficient
            n_step_max=rl_config_params.get('n_step_max', 50) # N-step returns
        )
        .resources(num_gpus=rl_config_params.get('num_gpus', 0)) # Set 0 if no GPUs, >0 if GPUs for NN training
        .debugging(log_level="INFO") # Set log level
        .build() # Build the algorithm configuration
    )

    # Ensure correct observation and action spaces are set (for debugging)
    # print(f"Observation space: {config.observation_space}")
    # print(f"Action space: {config.action_space}")

    # --- 5. Build and Train the Algorithm ---
    algo = config.build() # Build the algorithm instance

    # Setup TensorBoard logging to GCS
    # RLlib's Logger.log_dir is the base for TensorBoard event files
    # We'll set the result_dir (where Tune saves experiment results) to GCS
    from ray.tune.logger import pretty_print
    from ray.tune.experiment.trial import Trial

    # Configure Tune's logger to use the GCS bucket directly
    tune_config = {
        "num_samples": 1, # Only 1 "sample" for A3C if not hyperparameter tuning
        "config": config.to_dict(),
        "stop": {"training_iteration": args.train_iterations},
        "checkpoint_freq": rl_config_params.get('checkpoint_freq', 10), # Save checkpoint every X iterations
        "checkpoint_at_end": True,
        "local_dir": TENSORBOARD_LOG_DIR, # This is the key for GCS logging base
        "upload_dir": TENSORBOARD_LOG_DIR, # For syncing local_dir to cloud
        "loggers": [ray.tune.logger.DEFAULT_LOGGER, ray.tune.logger.JsonLogger] # Default loggers
    }

    print(f"TensorBoard logs will be saved to: {TENSORBOARD_LOG_DIR}")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")


    # Training loop
    print(f"\nStarting DRL training for {args.train_iterations} iterations...")
    for i in range(args.train_iterations):
        result = algo.train()

        # Print key metrics from the training result
        print(f"Iteration: {i+1}")
        print(pretty_print(result)) # Nicer print of results dict

        # RLlib automatically handles checkpointing to local_dir/upload_dir based on config.
        # We can also manually save more explicitly if needed.
        if (i + 1) % rl_config_params.get('checkpoint_freq', 10) == 0 or (i + 1) == args.train_iterations:
            # RLlib saves checkpoints to a subdirectory within its logging path.
            # The returned path is usually a local path on the head node.
            # For GCS, it gets synced by Tune's upload_dir.
            current_checkpoint_path = algo.save(checkpoint_dir=CHECKPOINT_DIR)
            print(f"Checkpoint saved in: {current_checkpoint_path}")

    print("\nDRL training complete.")

    # --- 6. Shut down Ray ---
    # Only shutdown if this script is responsible for managing the whole Ray lifecycle
    # If the Ray cluster is persistent, you might not want to shut it down here.
    # ray.shutdown() # Uncomment if you want to shutdown Ray client connection