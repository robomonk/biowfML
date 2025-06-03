import gym
import numpy as np
import ray
import uuid
from run_task import run_fastq_task

# --- Configuration ---
MAX_DURATION = 3600 # Example max task duration in seconds (e.g., 1 hour)
MAX_COST = 10.0     # Example max task cost
MAX_CPU = 4.0       # Example max CPU allocation
MAX_RAM = 16.0      # Example max RAM allocation in GB
GCS_BUCKET_NAME = "your-gcs-bucket-name"
GCS_INPUT_DIR = "fastq_inputs"
GCS_OUTPUT_DIR = "fastq_outputs"

class WorkflowEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(WorkflowEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3) # 0: Low, 1: Medium, 2: High
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.current_task_id = None
        self.last_task_result = None
        self.cpu_allocation_map = [1, 2, 4] # vCPU for actions 0, 1, 2
        self.ram_allocation_map = [4, 8, 16] # GB RAM for actions 0, 1, 2

    def reset(self):
        self.current_task_id = None
        self.last_task_result = None
        # Return initial observation (e.g., all zeros)
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action):
        cpu_allocation = self.cpu_allocation_map[action]
        ram_allocation = self.ram_allocation_map[action]
        task_id = str(uuid.uuid4())

        # Simulate getting a new FASTQ file (e.g., from a queue or GCS)
        fastq_gcs_path = f"gs://{GCS_BUCKET_NAME}/{GCS_INPUT_DIR}/sample.fastq.gz"
        output_gcs_dir = f"gs://{GCS_BUCKET_NAME}/{GCS_OUTPUT_DIR}/{task_id}"

        print(f"Scheduling task {task_id} with action {action} (CPU={cpu_allocation}, RAM={ram_allocation}GB)")

        # Launch the task asynchronously
        task_ref = run_fastq_task.remote(
            task_id,
            fastq_gcs_path,
            output_gcs_dir,
            cpu_allocation,
            ram_allocation
        )

        # Wait for the task to complete
        try:
            task_result = ray.get(task_ref)
            print(f"Task {task_id} completed: {task_result}")
        except Exception as e:
            print(f"Error running task {task_id}: {e}")
            task_result = {"task_id": task_id, "duration": -1, "cost": -1, "cpu_utilization": -1, "ram_utilization": -1, "error": str(e)}

        self.last_task_result = task_result
        self.current_task_id = task_id

        # Calculate reward
        if task_result["cost"] >= 0:
            reward = -task_result["cost"]
        else:
            reward = -100 # Penalize errors heavily

        # Calculate observation for the next step (based on completed task)
        observation = self._calculate_observation(task_result)

        # Determine if episode is done (simple case: one task per episode)
        done = True # Or based on a number of tasks or other criteria

        info = {
            "task_id": task_id,
            "cpu_allocation": cpu_allocation,
            "ram_allocation": ram_allocation,
            "result": task_result
        }

        return observation, reward, done, info

    def _calculate_observation(self, task_result):
        duration = task_result.get("duration", 0)
        cost = task_result.get("cost", 0)
        cpu_utilization = task_result.get("cpu_utilization", 0)
        ram_utilization = task_result.get("ram_utilization", 0)

        # Normalize values
        norm_duration = min(duration / MAX_DURATION, 1.0)
        norm_cost = min(cost / MAX_COST, 1.0)
        norm_cpu = min(cpu_utilization / MAX_CPU, 1.0)
        norm_ram = min(ram_utilization / MAX_RAM, 1.0)

        return np.array([norm_duration, norm_cost, norm_cpu, norm_ram], dtype=np.float32)

    def render(self, mode='human'):
        # Optional: Print current state or task info
        print(f"Current Task: {self.current_task_id}, Last Result: {self.last_task_result}")

    def close(self):
        # Clean up resources (e.g., Ray connections)
        pass
