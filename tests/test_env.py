"""
Unit and Integration Tests for Milestone 2: SongRecEnv Gymnasium Environment.
"""

import numpy as np
import pytest
import pandas as pd
from gymnasium.utils.env_checker import check_env

from src.data.ingestion import DatasetManager
from src.data.feature_store import FeatureStore
from src.env.response_model import UserResponseModel
from src.env.music_env import SongRecEnv, PlaybackEvent, compute_reward


@pytest.fixture(scope="module")
def env_setup():
    manager = DatasetManager(processed_dir="data/processed", seed=42)
    users_df = pd.read_parquet("data/processed/users.parquet")
    songs_df = pd.read_parquet("data/processed/songs.parquet")

    feature_store = FeatureStore(processed_dir="data/processed")
    feature_store.load_and_index(users_df, songs_df)

    resp_model = UserResponseModel(models_dir="models")
    resp_model.load()

    env = SongRecEnv(feature_store=feature_store, response_model=resp_model, max_session_len=20)
    return env


def test_gymnasium_check_env(env_setup):
    """Verifies that SongRecEnv passes standard Gymnasium API compliance checks."""
    check_env(env_setup.unwrapped, skip_render_check=True)


def test_reward_bounds():
    # 1. Best possible event: no skip (+0.5), liked (+1.0), saved (+0.5), novelty (+0.1) -> +2.1
    best_event = PlaybackEvent(
        song_id="s1", artist="ArtistA", genre="Jazz",
        skip_type="no_skip", liked=1, saved_to_playlist=1
    )
    r_best = compute_reward(best_event, [])
    assert r_best == pytest.approx(2.1, rel=1e-3)

    # 2. Worst possible event: early skip (-1.0), repeated artist (-0.3) -> -1.3
    prev_event = PlaybackEvent(
        song_id="s0", artist="ArtistA", genre="Pop",
        skip_type="no_skip", liked=0, saved_to_playlist=0
    )
    worst_event = PlaybackEvent(
        song_id="s1", artist="ArtistA", genre="Pop",
        skip_type="skip_early", liked=0, saved_to_playlist=0
    )
    r_worst = compute_reward(worst_event, [prev_event])
    assert r_worst == pytest.approx(-1.3, rel=1e-3)


def test_episode_reset_and_step(env_setup):
    obs, info = env_setup.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env_setup.state_dim,)
    assert not np.isnan(obs).any()

    # Step through 5 actions
    for _ in range(5):
        action = np.random.randint(0, env_setup.num_songs)
        obs, reward, terminated, truncated, step_info = env_setup.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "event" in step_info
