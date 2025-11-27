"""
Traditional CPU scheduling algorithms.
"""
from typing import List
from task import Task
import heapq


class BaseScheduler:
    """Base class for all schedulers."""

    def __init__(self):
        self.name = "BaseScheduler"
        self.current_time = 0.0
        self.completed_tasks = []
        self.running_task = None

    def schedule(self, tasks: List[Task]) -> List[Task]:
        """Schedule tasks and return completed tasks."""
        raise NotImplementedError

    def reset(self):
        """Reset scheduler state."""
        self.current_time = 0.0
        self.completed_tasks = []
        self.running_task = None


class FCFSScheduler(BaseScheduler):
    """First Come First Served (FCFS) scheduler."""

    def __init__(self):
        super().__init__()
        self.name = "FCFS"

    def schedule(self, tasks: List[Task]) -> List[Task]:
        sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
        self.current_time = 0.0
        self.completed_tasks = []

        for task in sorted_tasks:
            if self.current_time < task.arrival_time:
                self.current_time = task.arrival_time

            task.start_time = self.current_time
            task.completion_time = self.current_time + task.execution_time
            self.current_time = task.completion_time

            self.completed_tasks.append(task)

        return self.completed_tasks


class SJFScheduler(BaseScheduler):
    """Shortest Job First (non-preemptive)."""

    def __init__(self):
        super().__init__()
        self.name = "SJF"

    def schedule(self, tasks: List[Task]) -> List[Task]:
        sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
        self.current_time = 0.0
        self.completed_tasks = []
        ready_queue = []
        task_idx = 0

        while task_idx < len(sorted_tasks) or ready_queue:
            while task_idx < len(sorted_tasks) and sorted_tasks[task_idx].arrival_time <= self.current_time:
                task = sorted_tasks[task_idx]
                heapq.heappush(ready_queue, (task.execution_time, task.arrival_time, task))
                task_idx += 1

            if ready_queue:
                _, _, task = heapq.heappop(ready_queue)
                task.start_time = self.current_time
                task.completion_time = self.current_time + task.execution_time
                self.current_time = task.completion_time
                self.completed_tasks.append(task)
            else:
                if task_idx < len(sorted_tasks):
                    self.current_time = sorted_tasks[task_idx].arrival_time

        return self.completed_tasks


class PriorityScheduler(BaseScheduler):
    """Priority (lower number = higher priority)."""

    def __init__(self):
        super().__init__()
        self.name = "Priority"

    def schedule(self, tasks: List[Task]) -> List[Task]:
        sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
        self.current_time = 0.0
        self.completed_tasks = []
        ready_queue = []
        task_idx = 0

        while task_idx < len(sorted_tasks) or ready_queue:
            while task_idx < len(sorted_tasks) and sorted_tasks[task_idx].arrival_time <= self.current_time:
                task = sorted_tasks[task_idx]
                heapq.heappush(ready_queue, (task.priority, task.arrival_time, task))
                task_idx += 1

            if ready_queue:
                _, _, task = heapq.heappop(ready_queue)
                task.start_time = self.current_time
                task.completion_time = self.current_time + task.execution_time
                self.current_time = task.completion_time
                self.completed_tasks.append(task)
            else:
                if task_idx < len(sorted_tasks):
                    self.current_time = sorted_tasks[task_idx].arrival_time

        return self.completed_tasks


class RoundRobinScheduler(BaseScheduler):
    """Round Robin scheduler."""

    def __init__(self, time_quantum: float = 1.0):
        super().__init__()
        self.name = f"RoundRobin(q={time_quantum})"
        self.time_quantum = time_quantum

    def schedule(self, tasks: List[Task]) -> List[Task]:
        for task in tasks:
            task.remaining_time = task.execution_time
            task.start_time = None

        sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
        self.current_time = 0.0
        self.completed_tasks = []
        ready_queue: List[Task] = []
        task_idx = 0

        while task_idx < len(sorted_tasks) or ready_queue:
            while task_idx < len(sorted_tasks) and sorted_tasks[task_idx].arrival_time <= self.current_time:
                ready_queue.append(sorted_tasks[task_idx])
                task_idx += 1

            if ready_queue:
                task = ready_queue.pop(0)

                if task.start_time is None:
                    task.start_time = self.current_time

                execution_time = min(self.time_quantum, task.remaining_time)
                task.remaining_time -= execution_time
                self.current_time += execution_time

                if task.remaining_time <= 0:
                    task.completion_time = self.current_time
                    self.completed_tasks.append(task)
                else:
                    while task_idx < len(sorted_tasks) and sorted_tasks[task_idx].arrival_time <= self.current_time:
                        ready_queue.append(sorted_tasks[task_idx])
                        task_idx += 1
                    ready_queue.append(task)
            else:
                if task_idx < len(sorted_tasks):
                    self.current_time = sorted_tasks[task_idx].arrival_time

        return self.completed_tasks

