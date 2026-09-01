"""
Script to batch resolve and populate audio_url and image_url for catalog tracks from OpenSpot.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from src.data.openspot_client import openspot_client
from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_batch_resolve():
    songs_file = Path("data/processed/songs.parquet")
    users_file = Path("data/processed/users.parquet")
    models_dir = Path("models")

    df = pd.read_parquet(songs_file)
    logger.info(f"Loaded {len(df)} songs. Null audio_url count: {df['audio_url'].isna().sum()}")

    # Group by unique artists/titles to batch query efficiently
    resolved_count = 0
    for idx, row in df.iterrows():
        if pd.isna(row.get("audio_url")) or not row.get("audio_url"):
            title = row["title"]
            artist = row["artist_name"]
            q = f"{title} {artist}"
            try:
                tracks = openspot_client.search_songs(q, limit=1)
                if tracks:
                    df.at[idx, "audio_url"] = tracks[0].get("audio_url")
                    if pd.isna(row.get("image_url")) or "unsplash" in str(row.get("image_url", "")):
                        df.at[idx, "image_url"] = tracks[0].get("image_url")
                    df.at[idx, "album"] = tracks[0].get("album", title)
                    df.at[idx, "duration_formatted"] = tracks[0].get("duration_formatted", "3:30")
                    resolved_count += 1
            except Exception as e:
                logger.debug(f"Could not resolve '{q}': {e}")

        if idx % 100 == 0:
            logger.info(f"Progress: {idx}/{len(df)} (resolved {resolved_count})")

    logger.info(f"Batch resolution complete! Resolved {resolved_count} tracks.")
    df.to_parquet(songs_file, index=False)

    # Recompute embeddings
    users_df = pd.read_parquet(users_file)
    fs = FeatureStore(processed_dir="data/processed")
    fs.load_and_index(users_df, df)

    retrieval_model = TwoTowerModel(user_dim=fs.user_dim, song_dim=fs.song_dim, embed_dim=64)
    model_weight_path = models_dir / "baseline_two_tower.pt"
    if model_weight_path.exists():
        retrieval_model.load_state_dict(torch.load(str(model_weight_path), map_location="cpu"))
    retrieval_model.eval()

    with torch.no_grad():
        song_embs = retrieval_model.song_tower(fs.song_tensor).cpu().numpy()

    np.save(str(models_dir / "song_embeddings.npy"), song_embs)
    with open(str(models_dir / "song_ids.json"), "w") as f:
        json.dump(list(fs.song_id_to_idx.keys()), f)

    logger.info("Updated song embeddings and index files!")

if __name__ == "__main__":
    run_batch_resolve()
