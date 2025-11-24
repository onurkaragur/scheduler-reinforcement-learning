import random
import time

# --- Task/Job Management Utilities ---
class Job:
    """
    Represents a single schedulable task or job.
    """
    def __init__(self, job_id: int, duration_units: int, priority: int = 1, resource_needs: dict = None, deadline: float = None):
        """
        Initializes a job.
        param job_id: Unique identifier.
        param duration_units: Estimated execution time (in abstract time units).
        param priority: Importance (higher is more important).
        param resource_needs: Dict of {'resource_type': required_amount}.
        param deadline: Timestamp (float) by the which job should ideally finish. 
        """
        self.job_id = job_id
        self.duration_units = duration_units
        self.remaining_units = duration_units
        self.priority = priority
        self.resource_needs = resource_needs if resource_needs is not None else {"cpu": 1, "memory": 0.1}
        self.deadline = deadline
        self.arrival_time = time.time() # When the job was created or entered the queue.
        self.start_time = None
        self.finish_time = None
        self.is_running = False

        def execute(self, time_slice: int = 1) -> int:
            """
            Simulates executing the job for a time slice.
            """
            if not self.is_running: # Prevents meaningless executions.
                return 0
        
            executed = min(self.remaining_units, time_slice)
            self.remaining_units -= executed

            if self.remaining_units <= 0 and self.finish_time is None:
                self.finish_time = time.time()

            return executed

        def __repr__(self):
            status = "RUNNING" if self.is_running else "WAITING" if self.remaining_units > 0 else "DONE"
            return (f"Job(ID={self.job_id}), Prio={self.priority}, "
                    f"Rem={self.remaining_units}/{self.duration_units}, Status={status}")
        
class ResourcePool:
    """
    Manages available system resources.
    """
    def __init__(self, initial_resources: dict):
        self.capacity = initial_resources.copy()
        self.available = initial_resources.copy()

    def allocate(self, job_needs: dict) -> bool:
        """
        Attempts to allocate resources for a job.
        """
        can_allocate = True
        for res, amount in job_needs.items():
            if self.available.get(res, 0) < amount:
                can_allocate = False
                break

        if can_allocate:
            for res, amount in job_needs.items():
                self.available[res] -= amount
            return True
        return False
    
    def deallocate(self, job_needs: dict):
        """
        Releases resources used by a finished job.
        """ 
        for res, amount in job_needs.items():
            self.available[res] += amount 
