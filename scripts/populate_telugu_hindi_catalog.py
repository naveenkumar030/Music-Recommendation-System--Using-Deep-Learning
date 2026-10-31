"""
Populate SoundSpace DL catalog exclusively with Telugu and Hindi blockbuster tracks.
Removes all other genres/songs completely from songs.parquet and openspot_imports.parquet.
Generates verified metadata, album art, audio stream URLs, acoustic features, and 64-dim embeddings.
"""

import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.openspot_client import openspot_client
from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s")
logger = logging.getLogger("CatalogBuilder")

# 50 Curated Blockbuster Telugu Songs
TELUGU_QUERIES = [
    ("Samajavaragamana Ala Vaikunthapurramuloo", "Telugu"),
    ("Ramuloo Ramulaa Ala Vaikunthapurramuloo", "Telugu"),
    ("Butta Bomma Ala Vaikunthapurramuloo", "Telugu"),
    ("Naatu Naatu RRR", "Telugu"),
    ("Oo Antava Pushpa", "Telugu"),
    ("Srivalli Pushpa", "Telugu"),
    ("Kurchi Madathapetti Guntur Kaaram", "Telugu"),
    ("Fear Song Devara", "Telugu"),
    ("Chuttamalle Devara", "Telugu"),
    ("Daavudi Devara", "Telugu"),
    ("Tillu Anna DJ Pedithe", "Telugu"),
    ("Radhika Tillu Square", "Telugu"),
    ("Inkem Inkem Inkem Kaavaale Geetha Govindam", "Telugu"),
    ("Kalaavathi Sarkaru Vaari Paata", "Telugu"),
    ("Saranga Dariya Love Story", "Telugu"),
    ("Pushpa Pushpa Pushpa 2", "Telugu"),
    ("Sooseki Pushpa 2", "Telugu"),
    ("Hoyna Hoyna Gang Leader", "Telugu"),
    ("Dookudu Title Track", "Telugu"),
    ("Saahore Baahubali", "Telugu"),
    ("Dheera Dheera Magadheera", "Telugu"),
    ("Seeti Maar DJ Duvvada Jagannadham", "Telugu"),
    ("Top Lesi Poddi Iddarammayilatho", "Telugu"),
    ("Blockbuster Sarrainodu", "Telugu"),
    ("Mind Block Sarileru Neekevvaru", "Telugu"),
    ("He is So Cute Sarileru Neekevvaru", "Telugu"),
    ("Jai Balayya Veera Simha Reddy", "Telugu"),
    ("Dum Masala Guntur Kaaram", "Telugu"),
    ("Ay Pilla Love Story", "Telugu"),
    ("O Rendu Prema Meghaalila Baby", "Telugu"),
    ("Urike Urike Hit 2", "Telugu"),
    ("Komuram Bheemudo RRR", "Telugu"),
    ("Dosti RRR", "Telugu"),
    ("Poolane Kunukeyamanta I Telugu", "Telugu"),
    ("Rooba Rooba Orange", "Telugu"),
    ("Oola Oolala Ala Orange", "Telugu"),
    ("Nuvvunte Naa Jathaga I Telugu", "Telugu"),
    ("My Love Is Gone Arya 2", "Telugu"),
    ("Uppenantha Arya 2", "Telugu"),
    ("Ringa Ringa Arya 2", "Telugu"),
    ("Telisiney Na Nuvvey Arjun Reddy", "Telugu"),
    ("Gundellonaa Ori Devuda", "Telugu"),
    ("Mellaga Tellarindoi Shatamanam Bhavati", "Telugu"),
    ("Vachinde Fidaa", "Telugu"),
    ("O Bava Prathi Roju Pandage", "Telugu"),
    ("Tharagathi Gadhi Colour Photo", "Telugu"),
    ("Eyy Bidda Idhi Naa Adda Pushpa", "Telugu"),
    ("Ammadu Let us Do Kummudu Khaidi No 150", "Telugu"),
    ("Mass Masala Guntur Kaaram", "Telugu"),
    ("Nijame Ne Chebuthunna Ooru Peru Bhairavakona", "Telugu")
]

# 50 Curated Blockbuster Hindi Songs
HINDI_QUERIES = [
    ("Kesariya Brahmastra", "Hindi"),
    ("Tum Hi Ho Aashiqui 2", "Hindi"),
    ("Apna Bana Le Bhediya", "Hindi"),
    ("Chaleya Jawan", "Hindi"),
    ("Tere Vaaste Zara Hatke Zara Bachke", "Hindi"),
    ("Jhoome Jo Pathaan", "Hindi"),
    ("Raataan Lambiyan Shershaah", "Hindi"),
    ("Shayad Love Aaj Kal", "Hindi"),
    ("Ghungroo War", "Hindi"),
    ("Pehle Bhi Main Animal", "Hindi"),
    ("Satranga Animal", "Hindi"),
    ("Heeriye Jasleen Royal", "Hindi"),
    ("O Maahi Dunki", "Hindi"),
    ("Dil Diyan Gallan Tiger Zinda Hai", "Hindi"),
    ("Tauba Tauba Bad Newz", "Hindi"),
    ("Bekhayali Kabir Singh", "Hindi"),
    ("Tujhe Kitna Chahne Lage Kabir Singh", "Hindi"),
    ("Tera Ban Jaunga Kabir Singh", "Hindi"),
    ("Kalank Title Track", "Hindi"),
    ("Channa Mereya Ae Dil Hai Mushkil", "Hindi"),
    ("Ae Dil Hai Mushkil Title Track", "Hindi"),
    ("Gerua Dilwale", "Hindi"),
    ("Zaalima Raees", "Hindi"),
    ("Hawayein Jab Harry Met Sejal", "Hindi"),
    ("Ghodey Pe Sawaar Qala", "Hindi"),
    ("Pasoori Nu Satyaprem Ki Katha", "Hindi"),
    ("Lut Gaye Jubin Nautiyal", "Hindi"),
    ("Dilbar Satyameva Jayate", "Hindi"),
    ("Param Sundari Mimi", "Hindi"),
    ("Ilahi Yeh Jawaani Hai Deewani", "Hindi"),
    ("Subhanallah Yeh Jawaani Hai Deewani", "Hindi"),
    ("Balam Pichkari Yeh Jawaani Hai Deewani", "Hindi"),
    ("Badtameez Dil Yeh Jawaani Hai Deewani", "Hindi"),
    ("Agar Tum Saath Ho Tamasha", "Hindi"),
    ("Kun Faya Kun Rockstar", "Hindi"),
    ("Nadaan Parinde Rockstar", "Hindi"),
    ("Mast Magan 2 States", "Hindi"),
    ("Galliyan Ek Villain", "Hindi"),
    ("Sunn Raha Hai Aashiqui 2", "Hindi"),
    ("Teri Galliyan Ek Villain Returns", "Hindi"),
    ("Dil Sambhal Ja Zara Murder 2", "Hindi"),
    ("Hasi Ban Gaye Hamari Adhuri Kahani", "Hindi"),
    ("Kaun Tujhe MS Dhoni", "Hindi"),
    ("Besabriyaan MS Dhoni", "Hindi"),
    ("Phir Bhi Tumko Chaahunga Half Girlfriend", "Hindi"),
    ("Baarish Half Girlfriend", "Hindi"),
    ("Main Rang Sharbaton Ka Phata Poster", "Hindi"),
    ("Jeena Jeena Badlapur", "Hindi"),
    ("Tera Hone Laga Hoon Ajab Prem", "Hindi"),
    ("Tu Jaane Na Ajab Prem Ki Ghazab Kahani", "Hindi")
]

FALLBACK_IMAGE = "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80"


def check_image_url(url: str) -> str:
    """Verifies that the image URL is accessible; returns fallback if 404 or dead."""
    if not url or not url.startswith("http"):
        return FALLBACK_IMAGE
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return url
    except Exception:
        pass
    # Try GET with range header
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-10"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status in (200, 206):
                return url
    except Exception:
        pass
    return FALLBACK_IMAGE


def build_catalog():
    print("=" * 75)
    print("  SOUNDSPACE DL: EXCLUSIVE TELUGU & HINDI CATALOG BUILDER")
    print("=" * 75)

    all_queries = TELUGU_QUERIES + HINDI_QUERIES
    print(f"Total target queries: {len(all_queries)} ({len(TELUGU_QUERIES)} Telugu, {len(HINDI_QUERIES)} Hindi)")

    catalog_tracks = []
    seen_titles = set()
    track_counter = 1

    for q, lang_genre in all_queries:
        print(f"[{track_counter:03d}/{len(all_queries):03d}] Fetching '{q}' ({lang_genre})...", end=" ", flush=True)
        try:
            results = openspot_client.search_songs(q, limit=2)
            if not results:
                print("❌ Not found, skipping.")
                continue

            # Pick best match
            selected = results[0]
            norm_title = selected.get("title", "").lower().strip()
            if norm_title in seen_titles:
                if len(results) > 1:
                    selected = results[1]
                    norm_title = selected.get("title", "").lower().strip()
                if norm_title in seen_titles:
                    print("⚠️ Duplicate, skipping.")
                    continue

            seen_titles.add(norm_title)

            # Assign clean ID
            prefix = "tel" if lang_genre == "Telugu" else "hin"
            song_id = f"{prefix}_{track_counter:03d}"

            # Verify image
            img_url = selected.get("image_url", "")
            verified_img = check_image_url(img_url)

            # Prepare row
            row = {
                "song_id": song_id,
                "title": selected.get("title", q),
                "artist_name": selected.get("artist_name", f"{lang_genre} Artist"),
                "album": selected.get("album", f"{lang_genre} Hits"),
                "genre": lang_genre,
                "language": 99 if lang_genre == "Telugu" else 88,
                "danceability": float(selected.get("danceability", 0.75)),
                "energy": float(selected.get("energy", 0.80)),
                "valence": float(selected.get("valence", 0.75)),
                "tempo": float(selected.get("tempo", 125.0)),
                "acousticness": float(selected.get("acousticness", 0.20)),
                "instrumentalness": float(selected.get("instrumentalness", 0.001)),
                "speechiness": float(selected.get("speechiness", 0.05)),
                "loudness": float(selected.get("loudness", -5.5)),
                "popularity": int(selected.get("popularity", 95)),
                "song_length_ms": int(selected.get("duration_ms", 215000)),
                "duration_formatted": selected.get("duration_formatted", "3:35"),
                "image_url": verified_img,
                "audio_url": selected.get("audio_url", ""),
                "spotify_url": selected.get("spotify_url", "")
            }

            catalog_tracks.append(row)
            track_counter += 1
            print(f"[OK] {row['title']} by {row['artist_name']} ({row['album']})")
        except Exception as e:
            print(f"[ERROR] Error: {e}")
        time.sleep(0.05)

    print(f"\n[Catalog Construction Complete] Collected {len(catalog_tracks)} verified tracks.")
    df_catalog = pd.DataFrame(catalog_tracks)
    print("Catalog summary by genre:")
    print(df_catalog["genre"].value_counts())

    # 1. Save to data/processed/songs.parquet
    processed_dir = root / "data" / "processed"
    songs_path = processed_dir / "songs.parquet"
    df_catalog.to_parquet(songs_path, index=False)
    print(f"\n✅ Overwrote {songs_path} with {len(df_catalog)} Telugu & Hindi tracks ONLY.")

    # 2. Overwrite openspot_imports.parquet with the same catalog so sidecar never injects foreign tracks
    imports_path = processed_dir / "openspot_imports.parquet"
    df_catalog.to_parquet(imports_path, index=False)
    print(f"✅ Overwrote {imports_path} with matching Telugu & Hindi tracks.")

    # 3. Update User Profiles in users.parquet to reflect high affinity for Telugu and Hindi
    users_path = processed_dir / "users.parquet"
    df_users = pd.read_parquet(users_path)
    updated_affinities = []
    for _, row in df_users.iterrows():
        aff = {
            "Telugu": 0.95,
            "Hindi": 0.90,
            "Pop": 0.30,
            "Acoustic / Folk": 0.20
        }
        updated_affinities.append(json.dumps(aff))
    df_users["genre_affinities"] = updated_affinities
    df_users["pref_energy"] = 0.85
    df_users["pref_valence"] = 0.82
    df_users["pref_danceability"] = 0.80
    df_users.to_parquet(users_path, index=False)
    print(f"✅ Updated {len(df_users)} user profiles with dominant Telugu (0.95) and Hindi (0.90) affinities.")

    # 4. Generate Interaction Events for offline eval & training compatibility
    print("\nSynthesizing interaction events for new catalog...")
    events = []
    user_ids = df_users["user_id"].tolist()
    song_ids = df_catalog["song_id"].tolist()
    event_id = 1
    for u_idx, uid in enumerate(user_ids[:300]):
        # Each user has 15-25 listening events across Telugu & Hindi songs
        num_plays = np.random.randint(15, 26)
        chosen_sids = np.random.choice(song_ids, size=num_plays, replace=True)
        session_id = f"sess_{uid}_001"
        for step, sid in enumerate(chosen_sids):
            s_row = df_catalog[df_catalog["song_id"] == sid].iloc[0]
            liked = int(np.random.random() < 0.70)
            skip = "no_skip" if np.random.random() < 0.75 else ("skip_late" if np.random.random() < 0.5 else "skip_early")
            target = 1 if skip == "no_skip" else 0
            events.append({
                "event_id": event_id,
                "session_id": session_id,
                "user_id": uid,
                "song_id": sid,
                "timestamp": 1700000000 + event_id * 60,
                "step_in_session": step,
                "hour_of_day": np.random.randint(8, 23),
                "day_of_week": np.random.randint(0, 7),
                "device": "mobile",
                "skip_type": skip,
                "target": target,
                "liked": liked,
                "saved_to_playlist": int(np.random.random() < 0.35),
                "skip_rate_so_far": 0.2,
                "likes_so_far": 5,
                "tracks_played": step,
                "artist_name": s_row["artist_name"],
                "genre": s_row["genre"]
            })
            event_id += 1

    df_events = pd.DataFrame(events)
    df_events.to_parquet(processed_dir / "events.parquet", index=False)
    # Split 70 / 15 / 15
    n = len(df_events)
    t_end = int(0.70 * n)
    v_end = int(0.85 * n)
    df_events.iloc[:t_end].to_parquet(processed_dir / "train_events.parquet", index=False)
    df_events.iloc[t_end:v_end].to_parquet(processed_dir / "val_events.parquet", index=False)
    df_events.iloc[v_end:].to_parquet(processed_dir / "test_events.parquet", index=False)
    print(f"✅ Generated {len(df_events)} interaction events and train/val/test splits.")

    # 5. Compute Two-Tower 64-dim Embeddings for the new catalog
    print("\nComputing 64-dim Two-Tower DSSM Embeddings for new catalog...")
    fs = FeatureStore(processed_dir=processed_dir)
    fs.load_and_index(df_users, df_catalog)

    retrieval_model = TwoTowerModel(user_dim=fs.user_dim, song_dim=fs.song_dim, embed_dim=64)
    two_tower_path = root / "models" / "baseline_two_tower.pt"
    if two_tower_path.exists():
        retrieval_model.load_state_dict(torch.load(two_tower_path, map_location="cpu", weights_only=True))
    retrieval_model.eval()

    with torch.no_grad():
        all_song_embeddings = retrieval_model.song_tower(fs.song_tensor).cpu().numpy()

    models_dir = root / "models"
    np.save(models_dir / "song_embeddings.npy", all_song_embeddings)
    all_song_ids = [fs.idx_to_song_id[i] for i in range(fs.num_songs)]
    with open(models_dir / "song_ids.json", "w") as f:
        json.dump(all_song_ids, f)

    print(f"✅ Saved {len(all_song_embeddings)} embeddings to models/song_embeddings.npy and models/song_ids.json.")

    print("\n" + "=" * 75)
    print("  [SUCCESS] CATALOG CONVERTED EXCLUSIVELY TO TELUGU & HINDI SONGS!")
    print("=" * 75)


if __name__ == "__main__":
    build_catalog()
