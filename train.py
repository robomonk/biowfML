import ray
from ray.rllib.algorithms.a3c import A3CConfig
from workflowenv import WorkflowEnv
import os

# --- Configuration ---
RAY_ADDRESS = "auto" # Connect to existing Ray cluster (e.g., on GKE)
NUM_WORKERS = 4      # Number of Ray workers for training
TRAINING_ITERATIONS = 100
CHECKPOINT_DIR = "ray_checkpoint_checkpoints"

if __name__ == "__main__":
    # Initialize Ray
    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        print(f"Ray initialized with address: {ray.get_head_address()}")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        exit(1)

    # Configure RLlib A3C algorithm
    config = (
        A3CConfig()
        .environment(WorkflowEnv)
        .rollouts(num_rollout_workers=NUM_WORKERS)
        .framework("torch") # Or "tf2"
        .training(gamma=0.99, lr=0.0001)
        .resources(num_gpus=0) # Adjust if GPUs are needed for training
        .checkpointing(checkpoint_interval=10) # Save checkpoints every 10 iterations
        .build()
    )

    # Build the algorithm
    algo = config.build()

    # Training loop
    for i in range(TRAINING_ITERATIONS):
        result = algo.train()
        print(f"Iteration: {i}, Reward Mean: {result['episode_reward_mean']}")

        # Save checkpoint periodically
        if i % 10 == 0:
            checkpoint_dir = algo.save(CHECKPOINT_DIR)
            print(f"Checkpoint saved in directory: {checkpoint_dir}")

    # Shut down Ray
    ray.shutdown()