"""
ANN-based scheduler (supervised imitation of a heuristic).

This scheduler trains a small MLP to imitate a labeling policy ("priority" or "sjf").
It is efficient: collects training samples by rolling through training tasks once,
trains with minibatches and uses action-masking at inference to avoid invalid picks.
"""
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from task import Task
from scheduler_env import SchedulerEnv


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(128, 64)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANNScheduler:
    """Supervised ANN scheduler that imitates a heuristic policy.

    label_policy: 'priority' or 'sjf' (shortest job first)
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        state_size: int = 7,
        label_policy: str = "priority",
        device: Optional[str] = None,
    ):
        self.max_queue_size = max_queue_size
        self.state_size = state_size
        self.input_dim = max_queue_size * state_size + 6
        self.output_dim = max_queue_size
        self.label_policy = label_policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.model = MLP(self.input_dim, self.output_dim).to(self.device)

    def _label_from_ready(self, ready_queue: List[Task]) -> int:
        if not ready_queue:
            return 0
        if self.label_policy == "sjf":
            # pick shortest execution_time
            idx = int(np.argmin([t.execution_time for t in ready_queue]))
            return idx
        # default: priority (lower number = higher priority)
        idx = int(np.argmin([t.priority for t in ready_queue]))
        return idx

    def _collect_supervised_data(self, tasks: List[Task]) -> Tuple[np.ndarray, np.ndarray]:
        env = SchedulerEnv(tasks, max_queue_size=self.max_queue_size, state_size=self.state_size)
        state, _ = env.reset()
        X = []
        y = []
        done = False
        while True:
            # if no ready tasks and no running task may advance, but env.step expects action
            # We label based on current ready_queue without performing env.step if no ready tasks
            if len(env.ready_queue) == 0 and env.task_idx >= len(env.tasks) and env.running_task is None:
                break

            if len(env.ready_queue) > 0:
                label = self._label_from_ready(env.ready_queue)
                X.append(state.copy())
                y.append(label)
                # perform the action to advance
                _, _, terminated, truncated, _ = env.step(label)
                if terminated or truncated:
                    break
                state = env._get_state()
            else:
                # advance time by stepping with arbitrary action (will apply idle penalty)
                _, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
                state = env._get_state()

        if not X:
            return np.zeros((0, self.input_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)

    def fit(self, tasks: List[Task], epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
        X, y = self._collect_supervised_data(tasks)
        if X.shape[0] == 0:
            return

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                logits = self.model(bx)
                loss = loss_fn(logits, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * bx.size(0)
            # epoch loss printed minimally
            # print(f"ANNScheduler epoch {epoch+1}/{epochs} loss {total_loss / len(dataset):.4f}")

    def predict_action(self, state: np.ndarray, ready_queue_size: int) -> int:
        self.model.eval()
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            logits = self.model(s).squeeze(0).cpu().numpy()
            # Mask invalid actions
            if ready_queue_size < self.output_dim:
                logits[ready_queue_size:] = -1e9
            action = int(np.argmax(logits))
            return action

    def schedule(self, tasks: List[Task]) -> List[Task]:
        env = SchedulerEnv(tasks, max_queue_size=self.max_queue_size, state_size=self.state_size)
        state, _ = env.reset()
        done = False
        while True:
            if env.running_task is None and len(env.ready_queue) == 0 and env.task_idx >= len(env.tasks):
                break
            if len(env.ready_queue) > 0:
                action = self.predict_action(state, len(env.ready_queue))
            else:
                action = 0
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                break

        return env.get_completed_tasks()
