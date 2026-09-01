"""
Unit and Integration Tests for Milestone 1: Two-Tower Retrieval and Vector Index.
"""

import numpy as np
import pytest
import torch

from src.models.retrieval import TwoTowerModel, VectorIndex


def test_two_tower_forward_and_loss():
    user_dim = 16
    song_dim = 20
    embed_dim = 32
    batch_size = 8

    model = TwoTowerModel(user_dim=user_dim, song_dim=song_dim, embed_dim=embed_dim)

    u_features = torch.randn(batch_size, user_dim)
    s_features = torch.randn(batch_size, song_dim)

    # Test forward pass dot product
    scores = model(u_features, s_features)
    assert scores.shape == (batch_size,)
    assert not torch.isnan(scores).any()

    # Test InfoNCE loss computation
    loss = model.compute_loss(u_features, s_features)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    # Test backward pass
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def test_vector_index_retrieval_and_latency():
    num_songs = 1000
    embed_dim = 64
    song_embeddings = np.random.randn(num_songs, embed_dim).astype(np.float32)
    song_ids = [f"trk_{i}" for i in range(num_songs)]

    index = VectorIndex(embed_dim=embed_dim)
    index.build_index(song_embeddings, song_ids)

    query_vec = np.random.randn(embed_dim).astype(np.float32)
    top_candidates, top_scores, latency_ms = index.query(query_vec, top_k=500)

    assert len(top_candidates) == 500
    assert len(top_scores) == 500
    # Latency should be sub-50ms
    assert latency_ms < 50.0, f"Retrieval latency too slow: {latency_ms:.2f} ms"
    # Scores should be in descending order
    assert all(top_scores[i] >= top_scores[i+1] for i in range(len(top_scores)-1))
