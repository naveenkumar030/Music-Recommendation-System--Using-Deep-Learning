"""
Two-Tower Candidate Retrieval Model and Vector Index.

Implements:
- User/Session Tower: projects user profile + session context into d-dimensional latent space.
- Song Tower: projects acoustic dimensions + genre into d-dimensional latent space.
- Contrastive InfoNCE / Cosine loss training for implicit feedback.
- Fast Vector Index for Approximate Nearest Neighbor (ANN) candidate retrieval.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserSessionTower(nn.Module):
    """Encodes user static profile + session context into d-dimensional embedding."""

    def __init__(self, user_dim: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        raw_embed = self.net(user_features)
        return F.normalize(raw_embed, p=2, dim=-1)


class SongTower(nn.Module):
    """Encodes acoustic features + genre into d-dimensional embedding."""

    def __init__(self, song_dim: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(song_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, song_features: torch.Tensor) -> torch.Tensor:
        raw_embed = self.net(song_features)
        return F.normalize(raw_embed, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Full Two-Tower Retrieval Model with cosine similarity and InfoNCE / contrastive loss."""

    def __init__(self, user_dim: int, song_dim: int, embed_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.user_tower = UserSessionTower(user_dim=user_dim, embed_dim=embed_dim)
        self.song_tower = SongTower(song_dim=song_dim, embed_dim=embed_dim)
        self.temperature = temperature

    def forward(self, user_features: torch.Tensor, song_features: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_tower(user_features)  # (B, d)
        s_emb = self.song_tower(song_features)  # (B, d)
        # Dot product cosine similarity
        return torch.sum(u_emb * s_emb, dim=-1)

    def compute_loss(self, user_features: torch.Tensor, pos_song_features: torch.Tensor) -> torch.Tensor:
        """In-batch negative InfoNCE contrastive loss."""
        u_emb = self.user_tower(user_features)        # (B, d)
        s_emb = self.song_tower(pos_song_features)    # (B, d)

        # Cosine similarity matrix (B, B)
        sim_matrix = torch.matmul(u_emb, s_emb.T) / self.temperature
        labels = torch.arange(user_features.size(0), device=user_features.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class VectorIndex:
    """Fast Approximate / Exact Nearest Neighbor Index over Song Embeddings."""

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.song_embeddings: Optional[torch.Tensor] = None  # (N_songs, embed_dim)
        self.song_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}

    def build_index(self, song_embeddings: np.ndarray, song_ids: List[str]):
        """Builds index from normalized song embeddings."""
        self.song_ids = list(song_ids)
        self.id_to_idx = {sid: idx for idx, sid in enumerate(self.song_ids)}
        tensor_emb = torch.tensor(song_embeddings, dtype=torch.float32)
        # Ensure unit normalization for fast cosine dot product
        self.song_embeddings = F.normalize(tensor_emb, p=2, dim=-1)

    def add_item(self, song_id: str, embedding: np.ndarray):
        """Dynamically adds or updates a song vector embedding in the index."""
        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
        if emb_tensor.dim() == 1:
            emb_tensor = emb_tensor.unsqueeze(0)
        norm_emb = F.normalize(emb_tensor, p=2, dim=-1)

        if song_id in self.id_to_idx:
            idx = self.id_to_idx[song_id]
            self.song_embeddings[idx] = norm_emb.squeeze(0)
            return

        idx = len(self.song_ids)
        self.song_ids.append(song_id)
        self.id_to_idx[song_id] = idx
        if self.song_embeddings is not None:
            self.song_embeddings = torch.cat([self.song_embeddings, norm_emb], dim=0)
        else:
            self.song_embeddings = norm_emb

    def query(self, query_vector: np.ndarray, top_k: int = 500) -> Tuple[List[str], np.ndarray, float]:
        """
        Retrieves top-k nearest candidate songs for a given query vector.
        Returns:
            (top_song_ids, top_scores, latency_ms)
        """
        t0 = time.perf_counter()
        q_tensor = torch.tensor(query_vector, dtype=torch.float32)
        if q_tensor.dim() == 1:
            q_tensor = q_tensor.unsqueeze(0)
        q_norm = F.normalize(q_tensor, p=2, dim=-1)

        # Batch matrix multiplication: (1, d) x (d, N) -> (1, N)
        scores = torch.matmul(q_norm, self.song_embeddings.T).squeeze(0)
        
        k = min(top_k, len(self.song_ids))
        top_scores, top_indices = torch.topk(scores, k=k)
        
        top_song_ids = [self.song_ids[idx.item()] for idx in top_indices]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return top_song_ids, top_scores.detach().cpu().numpy(), latency_ms
