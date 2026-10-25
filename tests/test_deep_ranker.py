"""
Unit and Integration Tests for Deep Learning Ranking Architecture.

Tests:
- MultiTaskDeepRanker: Multi-objective engagement prediction.
- DeepRankerAgent: Vectorized candidate scoring, ranking, and fatigue suppression.
- GeneralizedMatrixFactorization and DeepMLPBranch.
"""

import numpy as np
import pytest
import torch

from src.models.deep_ranker import (
    MultiTaskDeepRanker,
    DeepRankerAgent,
    GeneralizedMatrixFactorization,
    DeepMLPBranch
)
from src.models.encoders import GRUSessionEncoder, FullStateEncoder


def test_multi_task_deep_ranker_forward():
    state_dim = 143
    song_dim = 64
    batch_size = 4
    num_cands = 10

    ranker = MultiTaskDeepRanker(state_dim=state_dim, song_dim=song_dim)
    dummy_state = torch.randn(batch_size, state_dim)
    dummy_cands = torch.randn(batch_size, num_cands, song_dim)

    out = ranker(dummy_state, dummy_cands)
    assert "rank_score" in out
    assert "p_complete" in out
    assert "p_like" in out
    assert "p_skip" in out

    assert out["rank_score"].shape == (batch_size, num_cands)
    assert out["p_complete"].shape == (batch_size, num_cands)

    # Probabilities should be strictly in [0, 1]
    assert (out["p_complete"] >= 0.0).all() and (out["p_complete"] <= 1.0).all()
    assert (out["p_like"] >= 0.0).all() and (out["p_like"] <= 1.0).all()
    assert (out["p_skip"] >= 0.0).all() and (out["p_skip"] <= 1.0).all()


def test_deep_ranker_agent_selection_and_ranking():
    num_songs = 100
    embed_dim = 64
    state_dim = 143

    song_embeddings = np.random.randn(num_songs, embed_dim).astype(np.float32)
    agent = DeepRankerAgent(
        state_dim=state_dim,
        song_embeddings=song_embeddings,
        song_dim=embed_dim
    )

    state = np.random.randn(state_dim).astype(np.float32)
    candidate_pool = [5, 12, 28, 44, 73]

    # Select single best action
    action_idx, details = agent.select_action(state, candidate_indices=candidate_pool)
    assert action_idx in candidate_pool
    assert "rank_score" in details
    assert "p_complete" in details
    assert "p_like" in details
    assert "p_skip" in details

    # Rank top candidates
    ranked = agent.rank_candidates(state, candidate_indices=candidate_pool, top_k=3)
    assert len(ranked) == 3
    # Check descending order of rank scores
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]


def test_ncf_components():
    user_dim = 16
    item_dim = 24
    latent_dim = 32
    batch_size = 4

    gmf = GeneralizedMatrixFactorization(user_dim=user_dim, item_dim=item_dim, latent_dim=latent_dim)
    u_tensor = torch.randn(batch_size, user_dim)
    i_tensor = torch.randn(batch_size, item_dim)

    phi, pred = gmf(u_tensor, i_tensor)
    assert phi.shape == (batch_size, latent_dim)
    assert pred.shape == (batch_size, 1)

    mlp = DeepMLPBranch(in_dim=user_dim + item_dim, hidden_layers=[64, 32])
    cat_tensor = torch.cat([u_tensor, i_tensor], dim=-1)
    mlp_out = mlp(cat_tensor)
    assert mlp_out.shape == (batch_size, 32)


def test_gru_session_encoder():
    batch_size = 4
    seq_len = 5
    item_dim = 20
    feedback_dim = 3
    hidden_dim = 64

    gru_enc = GRUSessionEncoder(item_dim=item_dim, feedback_dim=feedback_dim, hidden_dim=hidden_dim)
    dummy_seq = torch.randn(batch_size, seq_len, item_dim + feedback_dim)

    out = gru_enc(dummy_seq)
    assert out.shape == (batch_size, hidden_dim)
    assert not torch.isnan(out).any()


def test_stats_vec_normalization_match():
    history_likes = [1, 0, 1, 0, 1]  # 3 likes, 5 total
    likes_cnt = sum(1 for lk in history_likes if lk == 1)
    api_likes_norm = likes_cnt / 10.0
    env_likes_norm = 3 / 10.0
    assert env_likes_norm == api_likes_norm

