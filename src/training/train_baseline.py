"""
Baseline Two-Tower Retrieval Training Script.

Trains user and song towers using in-batch InfoNCE contrastive loss on positive listening interactions.
Evaluates on validation split and saves weights for candidate generation.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel, VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class InteractionDataset(Dataset):
    """PyTorch Dataset yielding (user_feature_vec, positive_song_feature_vec)."""

    def __init__(self, events_df: pd.DataFrame, feature_store: FeatureStore):
        # Filter for positive / satisfaction interactions (repeat target == 1 or no_skip)
        pos_df = events_df[events_df["skip_type"] == "no_skip"].copy()
        if len(pos_df) < 100:
            pos_df = events_df.copy()

        self.user_ids = pos_df["user_id"].tolist()
        self.song_ids = pos_df["song_id"].tolist()
        self.feature_store = feature_store

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u_id = self.user_ids[idx]
        s_id = self.song_ids[idx]
        u_vec = self.feature_store.get_user_vector_by_id(u_id)
        s_vec = self.feature_store.get_song_vector_by_id(s_id)
        return torch.tensor(u_vec, dtype=torch.float32), torch.tensor(s_vec, dtype=torch.float32)


def train_two_tower(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    embed_dim: int = 64,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 10
) -> TwoTowerModel:
    """Trains Two-Tower retrieval model and saves checkpoints."""
    p_dir = Path(processed_dir)
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    users_df = pd.read_parquet(p_dir / "users.parquet")
    songs_df = pd.read_parquet(p_dir / "songs.parquet")
    train_events = pd.read_parquet(p_dir / "train_events.parquet")
    val_events = pd.read_parquet(p_dir / "val_events.parquet")

    feature_store = FeatureStore(processed_dir=processed_dir)
    feature_store.load_and_index(users_df, songs_df)

    train_ds = InteractionDataset(train_events, feature_store)
    val_ds = InteractionDataset(val_events, feature_store)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TwoTowerModel(
        user_dim=feature_store.user_dim,
        song_dim=feature_store.song_dim,
        embed_dim=embed_dim
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info("Starting Two-Tower training: %d epochs, %d train batches...", epochs, len(train_loader))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for u_batch, s_batch in train_loader:
            optimizer.zero_grad()
            loss = model.compute_loss(u_batch, s_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for u_batch, s_batch in val_loader:
                loss = model.compute_loss(u_batch, s_batch)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        logger.info("Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), m_dir / "baseline_two_tower.pt")

    # Save metadata & build index
    model.load_state_dict(torch.load(m_dir / "baseline_two_tower.pt"))
    model.eval()
    
    with torch.no_grad():
        song_tensors = feature_store.song_tensor
        song_embeddings = model.song_tower(song_tensors).cpu().numpy()

    # Pre-build vector index and persist
    v_index = VectorIndex(embed_dim=embed_dim)
    v_index.build_index(song_embeddings, list(songs_df["song_id"]))
    
    np.save(m_dir / "song_embeddings.npy", song_embeddings)
    with open(m_dir / "song_ids.json", "w") as f:
        json.dump(list(songs_df["song_id"]), f)

    logger.info("Saved trained baseline model and song index to %s", m_dir)
    return model


if __name__ == "__main__":
    train_two_tower()
