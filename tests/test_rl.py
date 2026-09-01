"""
Unit and Integration Tests for Milestone 3: RL Policies and Encoders.
"""

import numpy as np
import pytest
import torch

from src.models.encoders import GRUSessionEncoder, FullStateEncoder
from src.models.actor_critic import WolpertingerActor, WolpertingerCritic, WolpertingerAgent
from src.models.dqn import DQNAgent, DuelingQNetwork


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


def test_wolpertinger_actor_and_critic():
    state_dim = 32
    action_embed_dim = 16
    batch_size = 4
    k_cand = 10

    actor = WolpertingerActor(state_dim=state_dim, action_embed_dim=action_embed_dim)
    critic = WolpertingerCritic(state_dim=state_dim, action_embed_dim=action_embed_dim)

    dummy_state = torch.randn(batch_size, state_dim)
    proto_act = actor(dummy_state)
    assert proto_act.shape == (batch_size, action_embed_dim)
    # Norm should be 1.0
    norms = torch.norm(proto_act, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # Test single-action critic scoring
    single_q = critic(dummy_state, proto_act)
    assert single_q.shape == (batch_size, 1)

    # Test multi-candidate critic scoring
    cand_actions = torch.randn(batch_size, k_cand, action_embed_dim)
    cand_q = critic(dummy_state, cand_actions)
    assert cand_q.shape == (batch_size, k_cand)


def test_wolpertinger_agent_step_and_train():
    num_songs = 50
    embed_dim = 16
    state_dim = 24
    song_embeddings = np.random.randn(num_songs, embed_dim).astype(np.float32)

    agent = WolpertingerAgent(
        state_dim=state_dim,
        song_embeddings=song_embeddings,
        action_embed_dim=embed_dim,
        k_candidates=10
    )

    state = np.random.randn(state_dim).astype(np.float32)
    action_idx, info = agent.select_action(state)

    assert 0 <= action_idx < num_songs
    assert "proto_action" in info
    assert "q_scores" in info

    # Populate replay buffer and test training step
    for _ in range(20):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, num_songs)
        r = float(np.random.randn())
        s_prime = np.random.randn(state_dim).astype(np.float32)
        done = bool(np.random.rand() < 0.1)
        agent.replay_buffer.push(s, a, r, s_prime, done)

    train_info = agent.train_step(batch_size=8)
    assert "critic_loss" in train_info
    assert "actor_loss" in train_info
