"""
Wolpertinger-Style Actor-Critic RL Architecture for Large Discrete Action Spaces.

References:
- Dulac-Arnold et al., "Deep Reinforcement Learning in Large Discrete Action Spaces" (2015)
- Google / YouTube / Spotify large-scale RecSys RL papers

Components:
- Actor Network: outputs continuous proto-action embedding in song feature space.
- k-NN Action Lookup: retrieves k nearest candidate song embeddings.
- Critic Network: scores candidate actions and selects argmax Q(s, a).
- Target Networks with Polyak soft updates.
- Replay Buffer for off-policy transition storage.
"""

from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WolpertingerActor(nn.Module):
    """Actor network outputting continuous proto-action in song embedding space."""

    def __init__(self, state_dim: int, action_embed_dim: int = 64, hidden_layers: List[int] = [256, 128]):
        super().__init__()
        layers = []
        curr_dim = state_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.05)
            ])
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, action_embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        proto_action = self.net(state)
        # Unit normalize proto-action embedding
        return F.normalize(proto_action, p=2, dim=-1)


class WolpertingerCritic(nn.Module):
    """Critic network Q(s, a) estimating expected return for state-action pair."""

    def __init__(self, state_dim: int, action_embed_dim: int = 64, hidden_layers: List[int] = [256, 128]):
        super().__init__()
        layers = []
        curr_dim = state_dim + action_embed_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU()
            ])
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        """
        state: (Batch, state_dim) or (Batch, 1, state_dim)
        action_embed: (Batch, action_dim) or (Batch, K, action_dim)
        """
        if action_embed.dim() == 3:
            B, K, A_dim = action_embed.shape
            state_expanded = state.unsqueeze(1).expand(B, K, -1)
            cat_input = torch.cat([state_expanded, action_embed], dim=-1)
            q_vals = self.net(cat_input)  # (B, K, 1)
            return q_vals.squeeze(-1)      # (B, K)
        else:
            cat_input = torch.cat([state, action_embed], dim=-1)
            return self.net(cat_input)     # (B, 1)


class ReplayBuffer:
    """Experience replay buffer for off-policy transition storage."""

    def __init__(self, capacity: int = 200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action_idx: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class WolpertingerAgent:
    """
    Wolpertinger Actor-Critic Re-ranker Policy.
    """

    def __init__(
        self,
        state_dim: int,
        song_embeddings: np.ndarray,
        action_embed_dim: int = 64,
        k_candidates: int = 50,
        gamma: float = 0.93,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_embed_dim = action_embed_dim
        self.k_candidates = k_candidates
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Pre-normalized song embeddings: (N_songs, action_embed_dim)
        song_tensor = torch.tensor(song_embeddings, dtype=torch.float32, device=self.device)
        self.song_embeddings = F.normalize(song_tensor, p=2, dim=-1)
        self.num_songs = self.song_embeddings.shape[0]

        # Networks
        self.actor = WolpertingerActor(state_dim, action_embed_dim).to(self.device)
        self.actor_target = WolpertingerActor(state_dim, action_embed_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = WolpertingerCritic(state_dim, action_embed_dim).to(self.device)
        self.critic_target = WolpertingerCritic(state_dim, action_embed_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=1e-4)

        self.replay_buffer = ReplayBuffer(capacity=200000)

    def add_song_embedding(self, song_emb: np.ndarray):
        """Dynamically adds an embedding vector for a newly imported song."""
        new_emb = torch.tensor(song_emb, dtype=torch.float32, device=self.device)
        if new_emb.dim() == 1:
            new_emb = new_emb.unsqueeze(0)
        norm_emb = F.normalize(new_emb, p=2, dim=-1)
        self.song_embeddings = torch.cat([self.song_embeddings, norm_emb], dim=0)
        self.num_songs = self.song_embeddings.shape[0]

    def select_action(
        self,
        state: np.ndarray,
        candidate_indices: Optional[List[int]] = None,
        exploration_noise: float = 0.0
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Selects best action using Wolpertinger policy:
        1. Proto-action from actor: a_proto = Actor(state)
        2. k-NN lookup among candidate songs: N_k(a_proto)
        3. Critic scores: a* = argmax_{a in N_k} Critic(state, a)
        """
        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            proto_act = self.actor(s_tensor)  # (1, d)

            if exploration_noise > 0.0:
                noise = torch.randn_like(proto_act) * exploration_noise
                proto_act = F.normalize(proto_act + noise, p=2, dim=-1)

            # Restrict lookup to candidate_indices if provided (from Two-Tower shortlist)
            if candidate_indices is not None and len(candidate_indices) > 0:
                cand_tensor = torch.tensor(candidate_indices, dtype=torch.int64, device=self.device)
                cand_embeddings = self.song_embeddings[cand_tensor]  # (K_sub, d)
                sims = torch.matmul(proto_act, cand_embeddings.T).squeeze(0)  # (K_sub,)
                k = min(self.k_candidates, len(candidate_indices))
                _, top_sub_idx = torch.topk(sims, k=k)
                chosen_candidate_indices = [candidate_indices[i.item()] for i in top_sub_idx]
            else:
                # Full catalog lookup
                sims = torch.matmul(proto_act, self.song_embeddings.T).squeeze(0)  # (N_songs,)
                k = min(self.k_candidates, self.num_songs)
                _, top_idx = torch.topk(sims, k=k)
                chosen_candidate_indices = [i.item() for i in top_idx]

            # Critic scores among the k candidates
            k_cand_tensor = torch.tensor(chosen_candidate_indices, dtype=torch.int64, device=self.device)
            k_embeddings = self.song_embeddings[k_cand_tensor].unsqueeze(0)  # (1, k, d)

            q_scores = self.critic(s_tensor, k_embeddings).squeeze(0)  # (k,)
            best_k_idx = torch.argmax(q_scores).item()
            best_action_idx = chosen_candidate_indices[best_k_idx]
            best_q_val = q_scores[best_k_idx].item()

        return best_action_idx, {
            "proto_action": proto_act.cpu().numpy()[0],
            "candidate_indices": chosen_candidate_indices,
            "q_scores": q_scores.cpu().numpy().tolist(),
            "best_q": best_q_val
        }

    def train_step(self, batch_size: int = 128) -> Dict[str, float]:
        """Performs one gradient update step for Critic and Actor."""
        if len(self.replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        s_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        s_next = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Action embeddings of observed actions
        act_embeddings = self.song_embeddings[a_t]  # (B, d)

        # 1. Critic Update:
        with torch.no_grad():
            next_proto = self.actor_target(s_next)  # (B, d)
            # Use target critic to evaluate target proto-action
            next_q = self.critic_target(s_next, next_proto)  # (B, 1)
            target_q = r_t + (1.0 - d_t) * self.gamma * next_q

        current_q = self.critic(s_t, act_embeddings)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 2. Actor Update (Policy Gradient: maximize Q(s, Actor(s))):
        pred_proto = self.actor(s_t)
        actor_loss = -self.critic(s_t, pred_proto).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 3. Soft Target Updates
        for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q": current_q.mean().item()
        }

    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pt", map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
