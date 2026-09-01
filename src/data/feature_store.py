"""
Feature Store for Song Embeddings, Metadata, and User Profile Lookups.

Provides:
- High-dimensional acoustic + categorical song embeddings.
- User static taste representations.
- Fast matrix operations and tensor lookups for Candidate Generation and RL state construction.
- Dynamic runtime ingestion of OpenSpot tracks into active recommendation indices.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch


class FeatureStore:
    """Central feature repository for song embeddings, user profiles, and metadata."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.users_df: Optional[pd.DataFrame] = None
        self.songs_df: Optional[pd.DataFrame] = None
        
        # Mapping dicts
        self.song_id_to_idx: Dict[str, int] = {}
        self.idx_to_song_id: Dict[int, str] = {}
        self.user_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.genre_to_idx: Dict[str, int] = {}
        
        # Precomputed feature matrices
        self.song_features_matrix: Optional[np.ndarray] = None  # (N_songs, d_song)
        self.user_features_matrix: Optional[np.ndarray] = None  # (N_users, d_user)
        self.song_tensor: Optional[torch.Tensor] = None
        self.song_metadata_dict: Dict[str, Dict] = {}
        
        # Audio feature columns
        self.continuous_feature_cols = [
            "danceability", "energy", "valence", "tempo", 
            "acousticness", "instrumentalness", "speechiness", 
            "loudness", "popularity"
        ]

    def compute_song_feature_vector(self, row: Dict) -> np.ndarray:
        """Computes continuous normalized feature vector + one-hot genre vector for a song."""
        tempo_min, tempo_max = 50.0, 220.0
        loud_min, loud_max = -35.0, 0.0
        
        tempo = float(row.get("tempo", 120.0))
        loudness = float(row.get("loudness", -8.0))
        pop = float(row.get("popularity", 50.0))

        cont = [
            float(row.get("danceability", 0.5)),
            float(row.get("energy", 0.5)),
            float(row.get("valence", 0.5)),
            float(np.clip((tempo - tempo_min) / (tempo_max - tempo_min), 0.0, 1.0)),
            float(row.get("acousticness", 0.5)),
            float(row.get("instrumentalness", 0.0)),
            float(row.get("speechiness", 0.05)),
            float(np.clip((loudness - loud_min) / (loud_max - loud_min), 0.0, 1.0)),
            float(np.clip(pop / 100.0, 0.0, 1.0))
        ]
        genre = row.get("genre", "Pop")
        genre_vec = [0.0] * len(self.genre_to_idx)
        if genre in self.genre_to_idx:
            genre_vec[self.genre_to_idx[genre]] = 1.0
        elif "Pop" in self.genre_to_idx:
            genre_vec[self.genre_to_idx["Pop"]] = 1.0

        return np.array(cont + genre_vec, dtype=np.float32)

    def load_and_index(self, users_df: pd.DataFrame, songs_df: pd.DataFrame):
        """Loads and precomputes dense embedding matrices for users and songs."""
        self.users_df = users_df.copy()
        self.songs_df = songs_df.copy()

        # Build IDs
        self.song_id_to_idx = {sid: idx for idx, sid in enumerate(self.songs_df["song_id"])}
        self.idx_to_song_id = {idx: sid for sid, idx in self.song_id_to_idx.items()}
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users_df["user_id"])}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        genres = sorted(self.songs_df["genre"].unique().tolist())
        self.genre_to_idx = {g: idx for idx, g in enumerate(genres)}

        # Build Song Feature Matrix (continuous normalized + one-hot genre)
        song_cont_features = []
        for _, row in self.songs_df.iterrows():
            song_cont_features.append(self.compute_song_feature_vector(row.to_dict()))

        self.song_features_matrix = np.array(song_cont_features, dtype=np.float32)
        self.song_tensor = torch.tensor(self.song_features_matrix, dtype=torch.float32)

        # Build User Static Feature Matrix (pref_energy, pref_valence, pref_danceability, age_norm, genre_affinities)
        user_features = []
        for _, row in self.users_df.iterrows():
            u_affinities = json.loads(row["genre_affinities"]) if isinstance(row["genre_affinities"], str) else row["genre_affinities"]
            genre_weights = [u_affinities.get(g, 0.0) for g in genres]
            u_vec = [
                float(row["pref_energy"]),
                float(row["pref_valence"]),
                float(row["pref_danceability"]),
                float(row["bd"]) / 70.0,
                1.0 if row["gender"] == "female" else (0.5 if row["gender"] == "male" else 0.0)
            ] + genre_weights
            user_features.append(u_vec)

        self.user_features_matrix = np.array(user_features, dtype=np.float32)

        # Cache metadata dicts for O(1) sub-millisecond retrieval
        self.song_metadata_dict = {
            row["song_id"]: row.to_dict() for _, row in self.songs_df.iterrows()
        }

    def add_song(self, song_dict: Dict) -> int:
        """Dynamically indexes a new OpenSpot track into the live FeatureStore in memory."""
        song_id = song_dict["song_id"]
        if song_id in self.song_id_to_idx:
            self.song_metadata_dict[song_id] = song_dict
            return self.song_id_to_idx[song_id]

        idx = len(self.song_id_to_idx)
        self.song_id_to_idx[song_id] = idx
        self.idx_to_song_id[idx] = song_id

        vec = self.compute_song_feature_vector(song_dict)
        if self.song_features_matrix is not None:
            self.song_features_matrix = np.vstack([self.song_features_matrix, vec])
            self.song_tensor = torch.tensor(self.song_features_matrix, dtype=torch.float32)
        else:
            self.song_features_matrix = np.array([vec], dtype=np.float32)
            self.song_tensor = torch.tensor(self.song_features_matrix, dtype=torch.float32)

        self.song_metadata_dict[song_id] = song_dict
        new_row = pd.DataFrame([song_dict])
        self.songs_df = pd.concat([self.songs_df, new_row], ignore_index=True)
        return idx

    def get_song_vector_by_id(self, song_id: str) -> np.ndarray:
        """Returns the dense normalized feature vector for a given song_id."""
        idx = self.song_id_to_idx[song_id]
        return self.song_features_matrix[idx]

    def get_song_vector_by_idx(self, idx: int) -> np.ndarray:
        """Returns the dense feature vector by matrix index."""
        return self.song_features_matrix[idx]

    def get_user_vector_by_id(self, user_id: str) -> np.ndarray:
        """Returns the static user taste vector."""
        idx = self.user_id_to_idx[user_id]
        return self.user_features_matrix[idx]

    def get_song_metadata(self, song_id: str) -> Dict:
        """Returns metadata dictionary for a given song."""
        if song_id in self.song_metadata_dict:
            return self.song_metadata_dict[song_id]
        row = self.songs_df[self.songs_df["song_id"] == song_id].iloc[0]
        return row.to_dict()

    @property
    def song_dim(self) -> int:
        return self.song_features_matrix.shape[1] if self.song_features_matrix is not None else 21

    @property
    def user_dim(self) -> int:
        return self.user_features_matrix.shape[1] if self.user_features_matrix is not None else 17

    @property
    def num_songs(self) -> int:
        return len(self.song_id_to_idx)

    @property
    def num_users(self) -> int:
        return len(self.user_id_to_idx)
