"""
Deep Q-Network (DQN) agent for task scheduling.
"""
from collections import deque
from typing import Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    """Feedforward network producing Q-values."""

    def __init__(self, state_size: int, action_size: int, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture: separates value and advantage streams."""

    def __init__(self, state_size: int, action_size: int, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        # Shared feature layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.feature_layers = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay_episodes: int = 300,
        memory_size: int = 50000,
        batch_size: int = 128,
        target_update_freq: int = 1000,
        use_dueling: bool = True,
        device: Optional[str] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = max(1, epsilon_decay_episodes)
        self.decay_counter = 0
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.use_dueling = use_dueling

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Use Dueling DQN if enabled, otherwise standard DQN
        if use_dueling:
            self.q_network = DuelingDQN(state_size, action_size).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        else:
            self.q_network = DQN(state_size, action_size).to(self.device)
            self.target_network = DQN(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.update_target_network()

    def update_target_network(self):
        """Synchronize target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def replay(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: use main network to select actions, target network to evaluate
        next_actions = self.q_network(next_states).argmax(dim=1)
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze().detach()
        
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def decay_epsilon(self):
        if self.decay_counter < self.epsilon_decay_steps:
            self.decay_counter += 1
            decay_ratio = self.decay_counter / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_min) * decay_ratio
        else:
            self.epsilon = self.epsilon_min

    def save(self, filepath: str):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)

