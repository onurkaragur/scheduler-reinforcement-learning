import numpy as np
import time
import random
from job import Job, ResourcePool, generate_random_job
from typing import Tuple, List

# --- RL Environment Definition (The World) ---

class SchedulerEnv:
    """
    The core Reinforcement Learning Environment (Gym-style).
    This handles the jobs, resources, state calculation, and reward computation.
    """
    def __init__(self, resource_config: dict, max_queue_size: int = 10, max_time_steps: int = 1000):
        self.resource_pool = ResourcePool(resource_config)
        self.job_queue: List[Job] = []
        self.running_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.max_queue_size = max_queue_size
        self.max_time_steps = max_time_steps
        self.current_time_step = 0
        self.next_job_id = 0
        self.action_space_size = 3 # 0: Run Highest Priority, 1: Run Oldest, 2: Do Nothing

    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state for a new episode.
        """
        self.job_queue = []
        self.running_jobs = []
        self.completed_jobs = []
        self.current_time_step = 0
        self.next_job_id = 0
        # Start with a few initial jobs
        for _ in range(3):
            self._add_new_job()
        
        return self._get_state()

    def _add_new_job(self):
        """
        Adds a new random job to the queue if capacity allows.
        """
        if len(self.job_queue) < self.max_queue_size:
            new_job = generate_random_job(self.next_job_id)
            self.job_queue.append(new_job)
            self.next_job_id += 1
    
    def _process_running_jobs(self, time_slice: int = 1):
        """
        Executes running jobs and handles completions/resource deallocation.
        """
        finished_now = []
        for job in self.running_jobs:
            job.execute(time_slice)
            if job.remaining_units <= 0:
                finished_now.append(job)
        
        # Move finished jobs and deallocate resources
        for job in finished_now:
            job.is_running = False
            self.running_jobs.remove(job)
            self.completed_jobs.append(job)
            self.resource_pool.deallocate(job.resource_needs)
            print(f"--- Job {job.job_id} COMPLETED at step {self.current_time_step} ---")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Takes an action and advances the environment by one time step.
        :param action: The discrete action chosen by the agent (0, 1, or 2).
        :return: new_state, reward, done, info
        """
        # 1. Execute current running jobs and handle completions.
        self._process_running_jobs()

        # 2. Agent's decision (Attempt to schedule a job).
        reward = 0.0
        scheduled_job: Job = None

        if action == 0 and self.job_queue: # Run the highest priority.
            job_to_run = max(self.job_queue, key=lambda j: j.priority)
        elif action == 1 and self.job_queue: # Run oldest job (First Come, First Served).
            job_to_run = min(self.job_queue, key=lambda j: j.arrival_time)
        else: # Do nothing or queue is empty.
            job_to_run = None

        if job_to_run and not job_to_run.is_running:
            if self.resource_pool.allocate(job_to_run.resource_needs):
                job_to_run.is_running = True
                job_to_run.start_time = time.time()
                self.job_queue.remove(job_to_run)
                self.running_jobs.append(job_to_run)
                reward += 10 # Positive reward for successful scheduling decision.
                scheduled_job = job_to_run
            else:
                reward -= 5 # Penalty for trying to schedule an impossible job.

        # 3. Calculate reward (Focus on minimizing job latency).
        # Penalizing for jobs waiting in the queue.
        if self.job_queue:
            total_waiting_latency = sum((time.time() - job.arrival_time) for job in self.job_queue)
            reward -= total_waiting_latency * 0.1 # Small penalty for waiting time.

        # Penalty for tardiness.
        for job in self.running_jobs:
            if job.deadline and job.deadline < time.time():
                reward -= 1.0

        # 4. Advance time.
        self.current_time_step += 1
        if random.random() < 0.3: # 30% chance new job arrives.
            self._add_new_job()

        # 5. Check if episode is done.
        done = self.current_time_step >= self.max_time_steps

        new_state = self._get_state()
        info = {"running_jobs_count": len(self.running_jobs), "scheduled_job_id": scheduled_job.job_id if scheduled_job else None}

        return new_state, reward, done, info 

    def _get_state():
        pass



