"""
Reinforcement Learning environment for task scheduling.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
from task import Task


class SchedulerEnv:
    """
    Custom environment for task scheduling.

    State: Features of ready queue tasks + current time
    Action: Which task to schedule next (index in ready queue)
    Reward: Negative waiting time to encourage fast scheduling
    """

    def __init__(
        self,
        tasks: List[Task],
        max_queue_size: int = 10,
        state_size: int = 7,
        reward_scale: float = 50.0,
        completion_bonus: float = 1.0,
        idle_penalty: float = 0.05,
        max_steps: Optional[int] = None,
    ):
        self.original_tasks = tasks.copy()
        self.tasks: List[Task] = []
        self.max_queue_size = max_queue_size
        self.state_size = state_size
        self.reward_scale = reward_scale
        self.completion_bonus = completion_bonus
        self.idle_penalty = idle_penalty
        self.base_max_steps = max_steps
        self.total_tasks = len(tasks)

        self.action_space_size = max_queue_size
        # Enhanced state: queue features + system metrics (queue depth, load, etc.).
        # _get_state appends 6 system-level features, so add 6 here.
        self.state_dim = max_queue_size * state_size + 6

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        self.tasks = []
        for task in self.original_tasks:
            self.tasks.append(
                Task(
                    task_id=task.task_id,
                    cpu_usage=task.cpu_usage,
                    ram_usage=task.ram_usage,
                    disk_io=task.disk_io,
                    network_io=task.network_io,
                    priority=task.priority,
                    vm_id=task.vm_id,
                    execution_time=task.execution_time,
                    arrival_time=task.arrival_time,
                )
            )

        self.tasks = sorted(self.tasks, key=lambda t: t.arrival_time)

        self.current_time = 0.0
        self.task_idx = 0
        self.ready_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.running_task: Optional[Task] = None
        self.running_task_end_time: Optional[float] = None
        self.step_count = 0
        self.max_steps = self.base_max_steps or max(100, len(self.tasks) * 2)

        self._update_ready_queue()

        return self._get_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0.0
        terminated = False
        truncated = False
        self.step_count += 1

        if self.running_task is not None and self.current_time >= self.running_task_end_time:
            self.running_task.completion_time = self.running_task_end_time
            self.completed_tasks.append(self.running_task)
            
            # Enhanced completion reward: bonus based on priority and waiting time
            priority_bonus = (4 - self.running_task.priority) * 0.2  # Higher priority = higher bonus
            completion_reward = self.completion_bonus + priority_bonus
            reward += completion_reward
            
            self.running_task = None
            self.running_task_end_time = None

        if self.running_task is None and len(self.ready_queue) > 0:
            action = action % len(self.ready_queue)

            selected_task = self.ready_queue.pop(action)
            waiting_time = self.current_time - selected_task.arrival_time
            
            # Enhanced waiting time penalty: non-linear with task priority consideration
            wait_penalty = np.tanh(waiting_time / max(1.0, self.reward_scale))
            priority_weight = (4 - selected_task.priority) / 3.0  # High priority tasks penalize more
            wait_penalty *= priority_weight
            reward -= wait_penalty

            selected_task.start_time = self.current_time
            self.running_task = selected_task
            self.running_task_end_time = self.current_time + selected_task.execution_time
        elif self.running_task is None:
            # Idle penalty varies with queue depth (penalize idle when work is available elsewhere)
            remaining_tasks = len(self.tasks) - self.task_idx
            queue_pressure = len(self.ready_queue) / max(1, self.max_queue_size)
            idle_penalty_scaled = self.idle_penalty * (1 + queue_pressure)
            reward -= idle_penalty_scaled

        next_event_time = float("inf")
        if self.running_task_end_time is not None:
            next_event_time = min(next_event_time, self.running_task_end_time)
        if self.task_idx < len(self.tasks):
            next_event_time = min(next_event_time, self.tasks[self.task_idx].arrival_time)

        if next_event_time == float("inf"):
            terminated = True
            self.current_time += 0.1
        else:
            self.current_time = next_event_time

        self._update_ready_queue()

        if len(self.completed_tasks) == len(self.tasks):
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        state = self._get_state()
        info = {
            "completed_tasks": len(self.completed_tasks),
            "ready_queue_size": len(self.ready_queue),
            "current_time": self.current_time,
        }

        return state, reward, terminated, truncated, info

    def _update_ready_queue(self):
        while self.task_idx < len(self.tasks):
            task = self.tasks[self.task_idx]
            if task.arrival_time <= self.current_time:
                self.ready_queue.append(task)
                self.task_idx += 1
            else:
                break

    def _get_state(self) -> np.ndarray:
        """Enhanced state representation with system metrics."""
        queue_features = []
        
        # Queue feature extraction
        for i in range(self.max_queue_size):
            if i < len(self.ready_queue):
                task = self.ready_queue[i]
                waiting_time = self.current_time - task.arrival_time
                features = [
                    task.cpu_usage / 100.0,
                    task.ram_usage / 20000.0,
                    task.disk_io / 100.0,
                    task.network_io / 100.0,
                    task.priority / 3.0,
                    task.execution_time / 10.0,
                    waiting_time / 10.0,
                ]
                queue_features.extend(features)
            else:
                queue_features.extend([0.0] * self.state_size)

        # System-level features for better context
        queue_depth = len(self.ready_queue) / max(1, self.max_queue_size)  # Queue utilization [0, 1]
        system_load = len(self.completed_tasks) / max(1, self.total_tasks)  # Progress ratio [0, 1]
        remaining_load = (self.total_tasks - self.task_idx) / max(1, self.total_tasks)  # Upcoming workload
        max_waiting = max(
            [self.current_time - t.arrival_time for t in self.ready_queue],
            default=0.0
        ) / max(1.0, self.reward_scale)  # Oldest waiting task normalized
        running_status = 1.0 if self.running_task is not None else 0.0  # Is CPU busy?
        
        state = queue_features + [
            self.current_time / 100.0,
            queue_depth,
            system_load,
            remaining_load,
            max_waiting,
            running_status,
        ]
        return np.array(state, dtype=np.float32)

    def get_completed_tasks(self) -> List[Task]:
        return self.completed_tasks.copy()

