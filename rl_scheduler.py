"""
RL-based scheduler that wraps a trained DQN agent.
"""
from typing import List
from task import Task
from scheduler_env import SchedulerEnv
from agent import DQNAgent


class RLScheduler:
    """Uses a trained DQN agent to choose the next task."""

    def __init__(
        self,
        agent: DQNAgent,
        max_queue_size: int = 10,
        state_size: int = 7,
        reward_scale: float = 50.0,
        completion_bonus: float = 1.0,
        idle_penalty: float = 0.05,
    ):
        self.agent = agent
        self.max_queue_size = max_queue_size
        self.state_size = state_size
        self.reward_scale = reward_scale
        self.completion_bonus = completion_bonus
        self.idle_penalty = idle_penalty
        self.name = "RL (DQN)"

    def schedule(self, tasks: List[Task]) -> List[Task]:
        env = SchedulerEnv(
            tasks,
            max_queue_size=self.max_queue_size,
            state_size=self.state_size,
            reward_scale=self.reward_scale,
            completion_bonus=self.completion_bonus,
            idle_penalty=self.idle_penalty,
            max_steps=len(tasks) * 2,
        )
        state, _ = env.reset()
        done = False

        while not done:
            action = self.agent.act(state, training=False)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated

        return env.get_completed_tasks()

