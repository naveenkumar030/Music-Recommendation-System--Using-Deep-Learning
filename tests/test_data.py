"""
Unit and Integration Tests for Milestone 0: Foundations & Data Ingestion.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.ingestion import DatasetManager, load_or_generate_dataset
from src.data.feature_store import FeatureStore


@pytest.fixture(scope="module")
def dataset_data(tmp_path_factory):
    """Generates a small test dataset in an isolated directory."""
    test_dir = Path("data/test_processed")
    test_dir.mkdir(parents=True, exist_ok=True)
    manager = DatasetManager(processed_dir="data/test_processed", seed=42)
    users_df, songs_df, events_df = manager.generate_calibrated_dataset(
        num_users=50,
        num_songs=120,
        num_sessions_per_user=4,
        avg_tracks_per_session=8
    )
    split_info = manager.save_datasets(users_df, songs_df, events_df)
    return users_df, songs_df, events_df, split_info


def test_schema_and_null_checks(dataset_data):
    users_df, songs_df, events_df, _ = dataset_data

    # Check nulls
    assert users_df.isnull().sum().sum() == 0, "Users table contains null values"
    assert songs_df.isnull().sum().sum() == 0, "Songs table contains null values"
    assert events_df.isnull().sum().sum() == 0, "Events table contains null values"

    # Check required columns
    expected_user_cols = {"user_id", "city", "bd", "gender", "pref_energy", "pref_valence", "pref_danceability", "genre_affinities"}
    assert expected_user_cols.issubset(set(users_df.columns))

    expected_song_cols = {"song_id", "title", "artist_name", "genre", "danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness", "speechiness", "loudness", "popularity"}
    assert expected_song_cols.issubset(set(songs_df.columns))

    expected_event_cols = {"event_id", "session_id", "user_id", "song_id", "timestamp", "step_in_session", "skip_type", "target", "liked", "saved_to_playlist"}
    assert expected_event_cols.issubset(set(events_df.columns))


def test_time_ordered_splits_no_leakage(dataset_data):
    _, _, _, split_info = dataset_data

    assert split_info["train_start"] <= split_info["train_cutoff"]
    assert split_info["train_cutoff"] <= split_info["val_cutoff"]
    assert split_info["val_cutoff"] <= split_info["test_end"]

    # Verify files exist and have no session overlap
    train_df = pd.read_parquet("data/test_processed/train_events.parquet")
    val_df = pd.read_parquet("data/test_processed/val_events.parquet")
    test_df = pd.read_parquet("data/test_processed/test_events.parquet")


    train_sessions = set(train_df["session_id"])
    val_sessions = set(val_df["session_id"])
    test_sessions = set(test_df["session_id"])

    assert len(train_sessions.intersection(val_sessions)) == 0, "Train and Val have overlapping sessions!"
    assert len(train_sessions.intersection(test_sessions)) == 0, "Train and Test have overlapping sessions!"
    assert len(val_sessions.intersection(test_sessions)) == 0, "Val and Test have overlapping sessions!"


def test_feature_store_embeddings(dataset_data):
    users_df, songs_df, _, _ = dataset_data

    fs = FeatureStore(processed_dir="data/processed")
    fs.load_and_index(users_df, songs_df)

    assert fs.num_songs == len(songs_df)
    assert fs.num_users == len(users_df)
    assert fs.song_dim > 10
    assert fs.user_dim > 5

    # Check vector lookup for first song
    first_song_id = songs_df.iloc[0]["song_id"]
    song_vec = fs.get_song_vector_by_id(first_song_id)
    assert isinstance(song_vec, np.ndarray)
    assert song_vec.shape[0] == fs.song_dim
    assert not np.isnan(song_vec).any()

    # Check vector lookup for first user
    first_user_id = users_df.iloc[0]["user_id"]
    user_vec = fs.get_user_vector_by_id(first_user_id)
    assert isinstance(user_vec, np.ndarray)
    assert user_vec.shape[0] == fs.user_dim
    assert not np.isnan(user_vec).any()
