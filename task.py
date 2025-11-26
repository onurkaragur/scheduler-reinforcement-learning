"""
Task/Job class to represent scheduling tasks.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """Represents a task/job in the scheduling system."""
    task_id: int
    cpu_usage: float  # CPU usage percentage
    ram_usage: float  # RAM usage in MB
    disk_io: float  # Disk I/O in MB/s
    network_io: float  # Network I/O in MB/s
    priority: int  # Priority level (1=highest, 3=lowest typically)
    vm_id: int  # Target VM ID
    execution_time: float  # Execution time in seconds
    arrival_time: float = 0.0  # When the task arrives
    start_time: Optional[float] = None  # When execution starts
    completion_time: Optional[float] = None  # When execution completes
    remaining_time: Optional[float] = None  # Remaining execution time (for RR)
    
    def __post_init__(self):
        """Initialize remaining_time to execution_time if not set."""
        if self.remaining_time is None:
            self.remaining_time = self.execution_time
    
    def get_waiting_time(self) -> float:
        """Calculate waiting time (time between arrival and start)."""
        if self.start_time is None:
            return 0.0
        return self.start_time - self.arrival_time
    
    def get_turnaround_time(self) -> float:
        """Calculate turnaround time (completion - arrival)."""
        if self.completion_time is None:
            return 0.0
        return self.completion_time - self.arrival_time
    
    def get_response_time(self) -> float:
        """Calculate response time (start - arrival)."""
        return self.get_waiting_time()
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.completion_time is not None
    
    def get_features(self) -> list:
        """Get feature vector for ML/RL models."""
        return [
            self.cpu_usage,
            self.ram_usage,
            self.disk_io,
            self.network_io,
            self.priority,
            self.execution_time
        ]
    
    def __str__(self):
        return f"Task(id={self.task_id}, exec_time={self.execution_time:.2f}s, priority={self.priority})"
    
    def __repr__(self):
        return self.__str__()

