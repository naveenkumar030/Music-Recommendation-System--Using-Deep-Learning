"""
Script to Enrich and Ingest OpenSpot Songs into SoundSpace Catalog.

Fetches real tracks, HQ album art, and direct 320kbps audio streams
using OpenSpot client, merges with catalog, and regenerates model vector embeddings.
"""

import json
import logging
from pathlib import Path
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import torch

from src.data.openspot_client import OpenSpotClient, openspot_client
from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GENRES_TO_FETCH = [
    "Pop", "Hip-Hop", "EDM / Electronic", "Rock", "R&B / Soul", 
    "Lo-Fi / Chillhop", "Jazz", "Classical", "Indie / Alternative", 
    "Metal", "Acoustic / Folk", "Synthwave"
]

SEARCH_QUERIES = [
    ("Levitating", "Pop"), ("Blinding Lights", "Pop"), ("Cruel Summer", "Pop"),
    ("Starboy", "Pop"), ("As It Was", "Pop"), ("Anti-Hero", "Pop"),
    ("HUMBLE.", "Hip-Hop"), ("Not Like Us", "Hip-Hop"), ("God's Plan", "Hip-Hop"),
    ("SICKO MODE", "Hip-Hop"), ("No Role Modelz", "Hip-Hop"), ("Sunflower", "Hip-Hop"),
    ("Wake Me Up", "EDM / Electronic"), ("Levels", "EDM / Electronic"), ("Summer", "EDM / Electronic"),
    ("Get Lucky", "EDM / Electronic"), ("One More Time", "EDM / Electronic"), ("Animals", "EDM / Electronic"),
    ("Bohemian Rhapsody", "Rock"), ("Do I Wanna Know?", "Rock"), ("Paint It, Black", "Rock"),
    ("Everlong", "Rock"), ("Yellow", "Rock"), ("Smells Like Teen Spirit", "Rock"),
    ("Kill Bill", "R&B / Soul"), ("Chanel", "R&B / Soul"), ("Bad Habit", "R&B / Soul"),
    ("Best Part", "R&B / Soul"), ("Texas Sun", "R&B / Soul"), ("Heartbreak Anniversary", "R&B / Soul"),
    ("Lofi hip hop beats", "Lo-Fi / Chillhop"), ("Aruarian Dance", "Lo-Fi / Chillhop"),
    ("Nujabes", "Lo-Fi / Chillhop"), ("Lofi rain study", "Lo-Fi / Chillhop"),
    ("So What", "Jazz"), ("Take Five", "Jazz"), ("Fly Me to the Moon", "Jazz"), ("Autumn Leaves", "Jazz"),
    ("Moonlight Sonata", "Classical"), ("Clair de Lune", "Classical"), ("Fur Elise", "Classical"),
    ("Experience Ludovico", "Classical"), ("The Less I Know The Better", "Indie / Alternative"),
    ("Space Song", "Indie / Alternative"), ("Somebody Else", "Indie / Alternative"),
    ("Master of Puppets", "Metal"), ("Chop Suey!", "Metal"), ("Du Hast", "Metal"),
    ("Skinny Love", "Acoustic / Folk"), ("Ho Hey", "Acoustic / Folk"), ("Riptide", "Acoustic / Folk"),
    ("Nightcall", "Synthwave"), ("The Midnight Sunset", "Synthwave"), ("Tech Noir", "Synthwave")
]


def run_ingestion():
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    songs_file = processed_dir / "songs.parquet"
    users_file = processed_dir / "users.parquet"

    logger.info("Loading existing catalog...")
    existing_songs_df = pd.read_parquet(songs_file) if songs_file.exists() else pd.DataFrame()
    users_df = pd.read_parquet(users_file)

    collected_tracks = []
    seen_titles = set()

    # 1. Fetch real tracks for curated search queries
    logger.info("Fetching OpenSpot tracks from JioSaavn API...")
    for query, genre in SEARCH_QUERIES:
        try:
            results = openspot_client.search_songs(query, limit=5)
            for track in results:
                t_key = f"{track['title'].lower()}_{track['artist_name'].lower()}"
                if t_key not in seen_titles:
                    seen_titles.add(t_key)
                    track["genre"] = genre  # align to intended genre category
                    collected_tracks.append(track)
        except Exception as e:
            logger.warning(f"Error fetching for query '{query}': {e}")

    # 2. Fetch top charts for each genre
    for g in GENRES_TO_FETCH:
        try:
            charts = openspot_client.get_top_charts(g, limit=15)
            for track in charts:
                t_key = f"{track['title'].lower()}_{track['artist_name'].lower()}"
                if t_key not in seen_titles:
                    seen_titles.add(t_key)
                    track["genre"] = g
                    collected_tracks.append(track)
        except Exception as e:
            logger.warning(f"Error fetching top charts for genre '{g}': {e}")

    logger.info(f"Retrieved {len(collected_tracks)} unique real OpenSpot songs!")

    # Enrich existing dataframe songs with matching artwork & audio URLs if possible
    openspot_lookup = {
        f"{t['title'].lower()}_{t['artist_name'].lower()}": t for t in collected_tracks
    }

    enriched_rows = []
    for _, row in existing_songs_df.iterrows():
        r_dict = row.to_dict()
        key = f"{r_dict['title'].lower()}_{r_dict['artist_name'].lower()}"
        
        if key in openspot_lookup:
            matched = openspot_lookup[key]
            r_dict["image_url"] = matched.get("image_url")
            r_dict["audio_url"] = matched.get("audio_url")
            r_dict["album"] = matched.get("album", r_dict.get("title"))
            r_dict["duration_formatted"] = matched.get("duration_formatted", "3:30")
        else:
            # Provide high quality fallback artwork and streams
            if "image_url" not in r_dict or not r_dict.get("image_url"):
                r_dict["image_url"] = f"https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80"
            if "audio_url" not in r_dict or not r_dict.get("audio_url"):
                # Use a sample audio stream or keep placeholder
                r_dict["audio_url"] = None
            if "album" not in r_dict:
                r_dict["album"] = r_dict.get("title")
            if "duration_formatted" not in r_dict:
                dur_ms = r_dict.get("song_length_ms", 210000)
                sec = int(dur_ms // 1000)
                r_dict["duration_formatted"] = f"{sec // 60}:{sec % 60:02d}"

        enriched_rows.append(r_dict)

    # Add new OpenSpot tracks with unique IDs
    next_id_num = 200000
    for track in collected_tracks:
        # Check if already in catalog
        exists = any(
            r["title"].lower() == track["title"].lower() and 
            r["artist_name"].lower() == track["artist_name"].lower()
            for r in enriched_rows
        )
        if not exists:
            new_row = {
                "song_id": f"trk_{next_id_num}",
                "title": track["title"],
                "artist_name": track["artist_name"],
                "album": track.get("album", track["title"]),
                "genre": track["genre"],
                "danceability": track["danceability"],
                "energy": track["energy"],
                "valence": track["valence"],
                "tempo": track["tempo"],
                "acousticness": track["acousticness"],
                "instrumentalness": track["instrumentalness"],
                "speechiness": track["speechiness"],
                "loudness": track["loudness"],
                "popularity": track["popularity"],
                "song_length_ms": track["song_length_ms"],
                "duration_formatted": track.get("duration_formatted", "3:30"),
                "image_url": track.get("image_url"),
                "audio_url": track.get("audio_url"),
                "spotify_url": track.get("spotify_url", f"https://open.spotify.com/search/{urllib.parse.quote(track['title'])}"),
                "language": 52
            }
            enriched_rows.append(new_row)
            next_id_num += 1

    # Build final DataFrame
    final_df = pd.DataFrame(enriched_rows)
    logger.info(f"Final enriched songs catalog shape: {final_df.shape}")
    final_df.to_parquet(songs_file, index=False)
    logger.info(f"Saved enriched catalog to {songs_file}")

    # Recompute FeatureStore & Two-Tower Song Embeddings
    logger.info("Recomputing FeatureStore and Two-Tower latent embeddings...")
    fs = FeatureStore(processed_dir=str(processed_dir))
    fs.load_and_index(users_df, final_df)

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

    logger.info(f"Successfully updated song embeddings ({song_embs.shape}) and song_ids.json!")


if __name__ == "__main__":
    run_ingestion()
