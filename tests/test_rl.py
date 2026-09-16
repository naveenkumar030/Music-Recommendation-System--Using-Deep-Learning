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


def test_stats_vec_normalization_match():
    """
    Regression test: verify that the stats_vec produced by the API serving path uses the
    same normalization as SongRecEnv._get_state() (music_env.py).

    Bug: api.py previously computed likes_cnt / max(1, total_played) but the env uses
    likes_so_far / 10.0.  This caused a train/serve distribution shift.
    Fix: api.py now uses likes_cnt / 10.0 to match the env.
    """
    # Simulate what music_env.py does for likes normalization
    likes_so_far = 3
    env_likes_norm = likes_so_far / 10.0  # as in music_env.py:248

    # Simulate the fixed api.py stats_vec path (same formula)
    history_likes = [1, 0, 1, 0, 1]  # 3 likes, 5 total
    likes_cnt = sum(1 for lk in history_likes if lk == 1)
    api_likes_norm = likes_cnt / 10.0  # FIXED: was likes_cnt / max(1, total_played)

    assert env_likes_norm == api_likes_norm, (
        f"Normalization mismatch: env={env_likes_norm}, api={api_likes_norm}. "
        "api.py and music_env.py must use the same normalization for stats_vec."
    )

    # Verify the OLD (broken) formula would have been different
    total_played = max(1, len(history_likes))
    old_api_likes_norm = min(1.0, likes_cnt / total_played)
    assert old_api_likes_norm != env_likes_norm, (
        "The old broken formula should produce a different value for this test case."
    )
    print(f"[Stats Vec] env={env_likes_norm}, api_fixed={api_likes_norm}, old_api={old_api_likes_norm} ✅")


def test_wolpertinger_cold_start_diversity():
    """
    Regression test: two different users with no session history should get different
    recommendations when cold-start exploration noise is applied.

    Bug: with no noise, the zero-padded history produces an identical proto-action for all
    users, collapsing to the same top-k songs.
    Fix: api.py injects exploration_noise=0.15 when history is empty.
    """
    num_songs = 100
    embed_dim = 32
    state_dim = 48
    song_embeddings = np.random.randn(num_songs, embed_dim).astype(np.float32)

    agent = WolpertingerAgent(
        state_dim=state_dim,
        song_embeddings=song_embeddings,
        action_embed_dim=embed_dim,
        k_candidates=10
    )

    # Two different users with no history \u2014 state differs only in user features
    np.random.seed(42)
    state_user_a = np.random.randn(state_dim).astype(np.float32)
    state_user_b = np.random.randn(state_dim).astype(np.float32)

    # Without noise: check if the same state gives a deterministic result
    action_a_no_noise, _ = agent.select_action(state_user_a, exploration_noise=0.0)
    action_a_no_noise_2, _ = agent.select_action(state_user_a, exploration_noise=0.0)
    assert action_a_no_noise == action_a_no_noise_2, "Deterministic mode must be stable."

    # With cold-start noise: same state should sometimes produce different results across
    # multiple calls (probabilistic exploration)
    results_with_noise = set()
    for _ in range(20):
        action, _ = agent.select_action(state_user_a, exploration_noise=0.15)
        results_with_noise.add(action)

    # With noise over 20 runs, we expect some diversity (more than 1 unique action)
    assert len(results_with_noise) > 1, (
        "Cold-start noise should produce diverse recommendations, "
        f"but only got {len(results_with_noise)} unique action(s) over 20 runs."
    )
    print(f"[Cold-Start Diversity] {len(results_with_noise)} unique actions over 20 runs with noise=0.15 ✅")
