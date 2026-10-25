"""
Reorders songs.parquet so that all Telugu/Tollywood songs appear at indices 0..N,
guaranteeing that any catalog view, default slice, or browsing endpoint serves Telugu first.
Also regenerates models/song_embeddings.npy and models/song_ids.json.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"
    models_dir = root / "models"

    songs_path = data_dir / "songs.parquet"
    df = pd.read_parquet(songs_path)

    # Separate Telugu songs and others
    is_telugu = (df["genre"] == "Telugu") | (df["language"] == 99)
    telugu_df = df[is_telugu].copy()
    other_df = df[~is_telugu].copy()

    print(f"Telugu songs count: {len(telugu_df)}")
    print(f"Other songs count: {len(other_df)}")

    # Put Telugu songs first
    reordered_df = pd.concat([telugu_df, other_df], ignore_index=True)
    reordered_df.to_parquet(songs_path, index=False)
    print(f"✅ Saved reordered songs.parquet with {len(reordered_df)} songs.")
    print("Top 5 songs now:")
    for i, r in reordered_df.head(5).iterrows():
        print(f"  [{i}] {r['title']} - {r['artist_name']} ({r['genre']})")

    # Re-initialize feature store
    print("\nRegenerating embeddings and IDs...")
    fs = FeatureStore(processed_dir=data_dir)
    users_df = pd.read_parquet(data_dir / "users.parquet")
    songs_df = pd.read_parquet(songs_path).replace({np.nan: None})
    fs.load_and_index(users_df, songs_df)

    # Compute Two-Tower embeddings
    two_tower = TwoTowerModel(user_dim=fs.user_dim, song_dim=fs.song_dim, embed_dim=64)
    model_path = models_dir / "baseline_two_tower.pt"
    if model_path.exists():
        two_tower.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    two_tower.eval()

    with torch.no_grad():
        all_embeddings = two_tower.song_tower(fs.song_tensor).cpu().numpy()

    np.save(models_dir / "song_embeddings.npy", all_embeddings)
    all_song_ids = [fs.idx_to_song_id[i] for i in range(fs.num_songs)]
    with open(models_dir / "song_ids.json", "w") as f:
        json.dump(all_song_ids, f)

    print(f"✅ Saved {len(all_embeddings)} embeddings to models/song_embeddings.npy.")
    print(f"✅ Saved {len(all_song_ids)} IDs to models/song_ids.json.")


if __name__ == "__main__":
    main()
