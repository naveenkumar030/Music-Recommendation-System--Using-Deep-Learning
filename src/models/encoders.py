"""
Session and State Encoders for RL Policies.

Provides:
- GRUSessionEncoder: 2-layer GRU encoding sequential (track_embedding, feedback_vector) pairs.
- StateTransformerEncoder: Multi-head attention session encoder alternative.
- FullStateEncoder: Integrates user taste embedding, GRU session encoding, and context.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUSessionEncoder(nn.Module):
    """2-layer GRU encoding recent sequence of (song_embedding, feedback) interactions."""

    def __init__(self, item_dim: int, feedback_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.input_dim = item_dim + feedback_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        """
        seq_tensor: (Batch, Sequence_Len, item_dim + feedback_dim)
        Returns: (Batch, hidden_dim) final hidden state
        """
        # gru output: (Batch, Seq_len, hidden_dim), h_n: (num_layers, Batch, hidden_dim)
        _, h_n = self.gru(seq_tensor)
        # Take the top layer final hidden state
        last_hidden = h_n[-1]  # (Batch, hidden_dim)
        return F.relu(self.proj(last_hidden))


class FullStateEncoder(nn.Module):
    """Combines user static embedding, GRU session representation, and context features."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        context_dim: int = 6,
        session_hidden_dim: int = 128,
        state_embed_dim: int = 128
    ):
        super().__init__()
        self.session_gru = GRUSessionEncoder(item_dim=item_dim, hidden_dim=session_hidden_dim)
        
        in_dim = user_dim + session_hidden_dim + context_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, state_embed_dim)
        )

    def forward(self, user_features: torch.Tensor, session_seq: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        user_features: (B, user_dim)
        session_seq: (B, seq_len, item_dim + 3)
        context_features: (B, context_dim)
        """
        sess_emb = self.session_gru(session_seq)
        combined = torch.cat([user_features, sess_emb, context_features], dim=-1)
        return self.fusion(combined)
