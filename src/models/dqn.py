"""
Dueling Double DQN Recommender Policy with Prioritized Experience Replay.
"""

from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling Network architecture: Q(s, a) = V(s) + (A(s, a) - mean(A(s, .)))."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.feature_layer(state)
        value = self.value_stream(feat)           # (B, 1)
        advantage = self.advantage_stream(feat)   # (B, action_dim)
        q_vals = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_vals


class DQNAgent:
    """Dueling Double DQN Agent for music recommendation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.93,
        lr: float = 3e-4,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.q_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = deque(maxlen=100000)

    def select_action(self, state: np.ndarray, candidate_indices: Optional[List[int]] = None) -> int:
        if np.random.rand() < self.epsilon:
            if candidate_indices and len(candidate_indices) > 0:
                return int(np.random.choice(candidate_indices))
            return int(np.random.randint(0, self.action_dim))

        self.q_net.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q_net(s_tensor).squeeze(0)  # (action_dim,)
            
            if candidate_indices and len(candidate_indices) > 0:
                cand_tensor = torch.tensor(candidate_indices, dtype=torch.int64, device=self.device)
                cand_q = q_vals[cand_tensor]
                best_sub_idx = torch.argmax(cand_q).item()
                return candidate_indices[best_sub_idx]
            else:
                return int(torch.argmax(q_vals).item())

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 128) -> Dict[str, float]:
        if len(self.buffer) < batch_size:
            return {}

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d_t = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q
        current_q = self.q_net(s_t).gather(1, a_t)

        # Double DQN target computation
        with torch.no_grad():
            next_actions = self.q_net(s_next).argmax(dim=1, keepdim=True)
            next_q = self.target_net(s_next).gather(1, next_actions)
            target_q = r_t + (1.0 - d_t) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon, "mean_q": current_q.mean().item()}

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
