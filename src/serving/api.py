"""
FastAPI Serving API for the RL Song Recommendation System.

Provides REST endpoints for:
- /api/health: health check and model loading status.
- /api/songs: song catalog query and audio feature lookup.
- /api/users: synthetic persona and user profile list.
- /api/recommend: next-track or 10-track slate recommendation (Two-Tower -> Wolpertinger RL -> Business Rules).
- /api/recommend/hybrid: Full hybrid pipeline (Content-Based + Collaborative + Context + Diversity + Exploration).
- /api/feedback: online feedback ingestion and behavior signal logging.
- /api/behavior/log: Granular user behavior event logging (listen_ms, replay, search_click, playlist_save).
- /api/behavior/profile/{user_id}: Serialized behavior profile with genre heatmap, acoustic centroid, mood.
- /api/simulate: live multi-session A/B simulation comparing models.
- /api/metrics: offline eval report and guardrails.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import urllib.request
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.data.feature_store import FeatureStore
from src.data.openspot_client import openspot_client
from src.data.lyrics_service import lyrics_service
from src.data.behavior_store import behavior_store, BehaviorStore
from src.env.response_model import UserResponseModel
from src.env.music_env import SongRecEnv, PlaybackEvent, compute_reward
from src.models.retrieval import TwoTowerModel, VectorIndex
from src.models.actor_critic import WolpertingerAgent
from src.models.hybrid_ranker import hybrid_ranker, infer_context_from_hour, get_context_profile

app = FastAPI(
    title="RL Song Recommender API",
    description="Production-grade RL recommendation service with Two-Tower candidate retrieval and Wolpertinger Actor-Critic re-ranking.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OpenSpotImportRequest(BaseModel):
    song_id: str
    title: str
    artist_name: str
    album: Optional[str] = ""
    genre: Optional[str] = "Pop"
    image_url: Optional[str] = ""
    audio_url: Optional[str] = ""
    duration_ms: Optional[int] = 210000
    duration_formatted: Optional[str] = "3:30"
    danceability: Optional[float] = 0.65
    energy: Optional[float] = 0.70
    valence: Optional[float] = 0.60
    tempo: Optional[float] = 120.0
    acousticness: Optional[float] = 0.25
    instrumentalness: Optional[float] = 0.001
    speechiness: Optional[float] = 0.05
    loudness: Optional[float] = -6.5
    popularity: Optional[int] = 75
    spotify_url: Optional[str] = ""

# Global runtime state
STATE: Dict[str, Any] = {
    "feature_store": None,
    "retrieval_model": None,
    "vector_index": None,
    "rl_agent": None,
    "response_model": None,
    "env": None,
    "active_sessions": {},
    "behavior_store": behavior_store,   # Singleton — shared across all requests
    "hybrid_ranker": hybrid_ranker,     # Singleton — weighted scoring / LightGBM
}


class RecommendRequest(BaseModel):
    user_id: str = Field(default="usr_00001")
    history_song_ids: List[str] = Field(default_factory=list)
    history_skip_types: List[str] = Field(default_factory=list)
    history_likes: List[int] = Field(default_factory=list)
    model_type: str = Field(default="wolpertinger")  # 'wolpertinger', 'baseline', 'random'
    slate_size: int = Field(default=1)               # 1 for next-track, k for slate


class HybridRecommendRequest(BaseModel):
    """Request body for the hybrid recommendation pipeline."""
    user_id: str = Field(default="usr_00001")
    history_song_ids: List[str] = Field(default_factory=list)
    history_skip_types: List[str] = Field(default_factory=list)
    history_likes: List[int] = Field(default_factory=list)
    slate_size: int = Field(default=10)
    context_type: Optional[str] = None             # auto-inferred if None
    listen_history_ms: Dict[str, int] = Field(default_factory=dict)  # {song_id: ms_listened}
    replay_song_ids: List[str] = Field(default_factory=list)
    search_history: List[str] = Field(default_factory=list)
    pull_openspot: bool = Field(default=True)       # whether to fetch fresh OpenSpot candidates
    explore_slots: int = Field(default=2)           # forced exploration picks
    diversity_window: int = Field(default=3)        # same-genre window for diversity


class BehaviorLogRequest(BaseModel):
    """Granular user behavior event payload."""
    user_id: str
    song_id: str
    event_type: str   # liked|disliked|no_skip|skip_early|skip_late|replay|playlist_save|search_click|listen_end
    title: str = ""
    artist_name: str = ""
    genre: str = "Pop"
    listen_ms: int = 0
    duration_ms: int = 210000
    energy: float = 0.65
    valence: float = 0.60
    danceability: float = 0.60
    tempo: float = 120.0
    search_query: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    user_id: str
    song_id: str
    action_type: str  # 'no_skip', 'skip_early', 'skip_late', 'liked', 'disliked', 'saved'
    listen_ms: int = 0


class SimulateRequest(BaseModel):
    user_id: Optional[str] = None
    num_sessions: int = Field(default=20)
    models_to_compare: List[str] = Field(default=["baseline", "wolpertinger", "random"])


@app.on_event("startup")
def startup_event():
    """Initializes models and datasets into memory. Gracefully handles missing model files."""
    processed_dir = "data/processed"
    models_dir = "models"

    try:
        users_df = pd.read_parquet(f"{processed_dir}/users.parquet")
        songs_df = pd.read_parquet(f"{processed_dir}/songs.parquet").replace({np.nan: None})

        fs = FeatureStore(processed_dir=processed_dir)
        fs.load_and_index(users_df, songs_df)
        STATE["feature_store"] = fs
        print(f"[Startup] FeatureStore loaded: {fs.num_songs} songs, {fs.num_users} users.")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load FeatureStore: {e}")
        return

    try:
        # Load Baseline Model
        retrieval_model = TwoTowerModel(user_dim=fs.user_dim, song_dim=fs.song_dim, embed_dim=64)
        retrieval_model.load_state_dict(torch.load(f"{models_dir}/baseline_two_tower.pt", map_location="cpu"))
        retrieval_model.eval()
        STATE["retrieval_model"] = retrieval_model
        print(f"[Startup] Two-Tower retrieval model loaded.")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load Two-Tower model: {e}")

    try:
        # Load Vector Index
        song_embeddings = np.load(f"{models_dir}/song_embeddings.npy")
        with open(f"{models_dir}/song_ids.json") as f:
            song_ids = json.load(f)
        v_index = VectorIndex(embed_dim=64)
        v_index.build_index(song_embeddings, song_ids)
        STATE["vector_index"] = v_index
        print(f"[Startup] Vector index built: {len(song_ids)} embeddings.")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load Vector Index: {e}")
        song_embeddings = None

    try:
        # Load User Response Model & Environment
        resp_model = UserResponseModel(models_dir=models_dir)
        resp_model.load()
        STATE["response_model"] = resp_model

        env = SongRecEnv(feature_store=fs, response_model=resp_model, max_session_len=50)
        STATE["env"] = env
        print(f"[Startup] SongRecEnv ready. State dim: {env.state_dim}")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load Response Model / Env: {e}")

    try:
        # Load Wolpertinger RL Agent
        if song_embeddings is not None and STATE["env"] is not None:
            agent = WolpertingerAgent(
                state_dim=STATE["env"].state_dim,
                song_embeddings=song_embeddings,
                action_embed_dim=64,
                k_candidates=50
            )
            agent.load(f"{models_dir}/wolpertinger_rl")
            STATE["rl_agent"] = agent
            print(f"[Startup] Wolpertinger RL agent loaded.")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load Wolpertinger RL agent: {e}")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "feature_store": STATE["feature_store"] is not None,
            "retrieval_model": STATE["retrieval_model"] is not None,
            "vector_index": STATE["vector_index"] is not None,
            "rl_agent": STATE["rl_agent"] is not None,
        },
        "catalog_size": STATE["feature_store"].num_songs if STATE["feature_store"] else 0
    }


@app.get("/api/songs")
def get_songs(genre: Optional[str] = None, search: Optional[str] = None, limit: int = 50):
    fs: FeatureStore = STATE["feature_store"]
    df = fs.songs_df.copy()
    if genre and genre != "All":
        df = df[df["genre"] == genre]
    if search:
        s_lower = search.lower()
        df = df[df["title"].str.lower().str.contains(s_lower) | 
                df["artist_name"].str.lower().str.contains(s_lower) |
                df["song_id"].str.lower().str.contains(s_lower)]
    # Replace NaN with None for clean JSON serialization
    records = df.head(limit).replace({np.nan: None}).to_dict(orient="records")
    return records


@app.get("/api/openspot/search")
def openspot_search(q: str, limit: int = 20, p: int = 1):
    """Searches live OpenSpot catalog with real 320kbps audio streams and 500x500 album art."""
    if not q or not q.strip():
        return {"results": [], "count": 0}
    results = openspot_client.search_songs(q.strip(), limit=limit, p=p)
    return {"results": results, "count": len(results)}


@app.get("/api/openspot/autocomplete")
def openspot_autocomplete(q: str):
    """Instant auto-complete lookup for songs, artists, and albums."""
    if not q or not q.strip():
        return {"songs": [], "albums": [], "artists": []}
    return openspot_client.search_autocomplete(q.strip())


@app.get("/api/openspot/charts")
def openspot_charts(genre: str = "Pop", limit: int = 20):
    """Fetches trending genre charts from OpenSpot."""
    results = openspot_client.get_top_charts(genre=genre, limit=limit)
    return {"genre": genre, "results": results, "count": len(results)}


@app.post("/api/openspot/import")
def openspot_import_song(req: OpenSpotImportRequest):
    """
    Dynamically imports an OpenSpot track into the live FeatureStore,
    encodes its latent vector using Two-Tower SongTower, adds it to the VectorIndex,
    and updates the Wolpertinger RL Agent's action space in real time.
    """
    fs: FeatureStore = STATE["feature_store"]
    retrieval_model: TwoTowerModel = STATE["retrieval_model"]
    v_index: VectorIndex = STATE["vector_index"]
    agent: WolpertingerAgent = STATE["rl_agent"]

    if not fs or not retrieval_model or not v_index:
        raise HTTPException(status_code=503, detail="Server models not initialized.")

    track_dict = req.dict()
    song_id = req.song_id

    # 1. Index into FeatureStore
    idx = fs.add_song(track_dict)

    # 2. Encode with Song Tower
    s_vec = fs.get_song_vector_by_id(song_id)
    with torch.no_grad():
        s_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
        s_emb = retrieval_model.song_tower(s_tensor).squeeze(0).cpu().numpy()

    # 3. Add to Vector Index
    v_index.add_item(song_id, s_emb)

    # 4. Add to Wolpertinger Agent candidate space
    if agent is not None:
        agent.add_song_embedding(s_emb)

    return {
        "status": "success",
        "message": f"Successfully imported '{req.title}' by {req.artist_name} into RL catalog!",
        "song": fs.get_song_metadata(song_id),
        "catalog_size": fs.num_songs
    }


@app.get("/api/stream/resolve")
def resolve_song_stream(song_id: str):
    """Resolves and caches the direct 320kbps OpenSpot audio stream URL for any song."""
    fs: FeatureStore = STATE["feature_store"]
    if not fs:
        raise HTTPException(status_code=503, detail="Feature store not initialized.")

    if song_id not in fs.song_id_to_idx:
        raise HTTPException(status_code=404, detail="Song not found.")

    meta = fs.get_song_metadata(song_id)
    if meta.get("audio_url"):
        return {
            "song_id": song_id,
            "audio_url": meta["audio_url"],
            "image_url": meta.get("image_url"),
            "resolved": False
        }

    # Search OpenSpot for this title & artist
    q = f"{meta['title']} {meta['artist_name']}"
    tracks = openspot_client.search_songs(q, limit=2)
    if not tracks:
        tracks = openspot_client.search_songs(meta["title"], limit=2)

    if tracks:
        t = tracks[0]
        meta["audio_url"] = t.get("audio_url")
        if not meta.get("image_url") or "unsplash" in str(meta.get("image_url", "")):
            meta["image_url"] = t.get("image_url")
        fs.song_metadata_dict[song_id] = meta
        return {
            "song_id": song_id,
            "audio_url": meta["audio_url"],
            "image_url": meta.get("image_url"),
            "resolved": True
        }

    return {"song_id": song_id, "audio_url": None, "image_url": meta.get("image_url"), "resolved": False}


@app.get("/api/stream/proxy")
def proxy_audio_stream(url: str):
    """Proxies audio stream with full CORS and audio/mp4 headers to bypass any browser restrictions."""
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid audio stream URL.")

    def iter_stream():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iter_stream(),
        media_type="audio/mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.get("/api/lyrics")
def get_song_lyrics(
    song_id: Optional[str] = None,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    duration_ms: Optional[int] = None,
    genre: Optional[str] = None,
    energy: Optional[float] = None,
    valence: Optional[float] = None,
    tempo: Optional[float] = None
):
    """
    Returns synchronized LRC timestamped lyrics and plain text lyrics for any song.
    Integrates LRCLIB verified karaoke lyrics, JioSaavn lyrics, and calibrated procedural lyrics.
    """
    fs: FeatureStore = STATE["feature_store"]
    
    # Enrich from FeatureStore metadata if song_id is provided
    if song_id and fs and song_id in fs.song_id_to_idx:
        meta = fs.get_song_metadata(song_id)
        if not title:
            title = meta.get("title")
        if not artist:
            artist = meta.get("artist_name")
        if duration_ms is None:
            duration_ms = meta.get("duration_ms") or meta.get("song_length_ms")
        if not genre:
            genre = meta.get("genre")
        if energy is None:
            energy = meta.get("energy")
        if valence is None:
            valence = meta.get("valence")
        if tempo is None:
            tempo = meta.get("tempo")

    if not title:
        title = "Unknown Song"
    if not artist:
        artist = "Unknown Artist"

    lyrics_data = lyrics_service.get_lyrics(
        song_id=song_id or f"temp_{hash(title + artist)}",
        title=title,
        artist=artist,
        duration_ms=duration_ms or 210000,
        genre=genre or "Pop",
        energy=float(energy) if energy is not None else 0.65,
        valence=float(valence) if valence is not None else 0.60,
        tempo=float(tempo) if tempo is not None else 120.0
    )

    return lyrics_data


@app.get("/api/songs/{song_id}/related")
def get_related_songs(song_id: str, limit_per_section: int = 8):
    """
    Computes vector embedding cosine similarity and acoustic deltas
    to recommend related songs across multiple curated sections:
    - Acoustic Twins (overall nearest neighbors in latent acoustic space)
    - Same Genre (in-genre depth recommendations)
    - Cross-Genre Discovery (bridge tracks with similar mood/energy from other genres)
    - Vibe & Energy Transitions (boosted energy/tempo recommendations)
    """
    fs: FeatureStore = STATE["feature_store"]
    v_index: VectorIndex = STATE["vector_index"]

    if not fs or not v_index:
        raise HTTPException(status_code=503, detail="System models not initialized.")

    if song_id not in fs.song_id_to_idx:
        raise HTTPException(status_code=404, detail=f"Song ID '{song_id}' not found.")

    seed_meta = fs.get_song_metadata(song_id)
    seed_idx = v_index.id_to_idx.get(song_id)
    if seed_idx is None:
        raise HTTPException(status_code=404, detail="Song vector not found in index.")

    t0 = time.perf_counter()

    # Query vector similarity across all catalog songs
    seed_emb = v_index.song_embeddings[seed_idx]
    sims = torch.matmul(v_index.song_embeddings, seed_emb).squeeze().cpu().numpy()

    # Prepare ranked items
    ranked_indices = np.argsort(-sims)
    
    seed_genre = seed_meta["genre"]
    seed_energy = float(seed_meta["energy"])
    seed_valence = float(seed_meta["valence"])
    seed_tempo = float(seed_meta["tempo"])
    seed_dance = float(seed_meta["danceability"])

    acoustic_twins = []
    same_genre_songs = []
    cross_genre_songs = []
    vibe_transitions = []
    # Track which song_ids have been assigned to any section (for deduplication)
    seen_per_section: dict = {
        "acoustic_twins": set(),
        "same_genre": set(),
        "cross_genre": set(),
        "vibe_transitions": set(),
    }

    # Minimum delta thresholds for meaningful vibe transitions
    VIBE_MIN_ENERGY_DELTA = 0.05
    VIBE_MIN_TEMPO_DELTA = 5.0

    for idx in ranked_indices:
        cand_sid = v_index.song_ids[idx]
        if cand_sid == song_id:
            continue
        
        sim_val = float(sims[idx])
        meta = fs.get_song_metadata(cand_sid)

        # Normalize cosine similarity from [-1, 1] to [0%, 100%] correctly
        match_pct = round(max(0.0, min(99.9, (sim_val + 1.0) / 2.0 * 100.0)), 1)
        
        c_energy = float(meta["energy"])
        c_valence = float(meta["valence"])
        c_tempo = float(meta["tempo"])
        c_dance = float(meta["danceability"])
        c_genre = meta["genre"]

        energy_diff = round(c_energy - seed_energy, 2)
        valence_diff = round(c_valence - seed_valence, 2)
        tempo_diff = round(c_tempo - seed_tempo, 1)
        dance_diff = round(c_dance - seed_dance, 2)

        song_payload = {
            **meta,
            "match_score": round(sim_val, 4),
            "match_percentage": match_pct,
            "acoustic_deltas": {
                "energy_diff": energy_diff,
                "valence_diff": valence_diff,
                "tempo_diff": tempo_diff,
                "dance_diff": dance_diff
            }
        }

        # 1. Acoustic Twins (Overall Top Nearest Neighbors)
        if cand_sid not in seen_per_section["acoustic_twins"] and len(acoustic_twins) < limit_per_section:
            acoustic_twins.append(song_payload)
            seen_per_section["acoustic_twins"].add(cand_sid)

        # 2. Same Genre Relatives (deduplicated within section)
        if c_genre == seed_genre and cand_sid not in seen_per_section["same_genre"] and len(same_genre_songs) < limit_per_section:
            same_genre_songs.append(song_payload)
            seen_per_section["same_genre"].add(cand_sid)

        # 3. Cross-Genre Discovery (deduplicated within section)
        if c_genre != seed_genre and cand_sid not in seen_per_section["cross_genre"] and len(cross_genre_songs) < limit_per_section:
            cross_genre_songs.append(song_payload)
            seen_per_section["cross_genre"].add(cand_sid)

        # 4. Vibe / Energy Transitions — require meaningful minimum delta
        energy_up = energy_diff >= VIBE_MIN_ENERGY_DELTA
        tempo_up = tempo_diff >= VIBE_MIN_TEMPO_DELTA
        if (energy_up or tempo_up) and cand_sid not in seen_per_section["vibe_transitions"] and len(vibe_transitions) < limit_per_section:
            vibe_transitions.append(song_payload)
            seen_per_section["vibe_transitions"].add(cand_sid)

        # Early break once all sections are full
        if (len(acoustic_twins) >= limit_per_section and 
            len(same_genre_songs) >= limit_per_section and 
            len(cross_genre_songs) >= limit_per_section and 
            len(vibe_transitions) >= limit_per_section):
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "seed_song": seed_meta,
        "sections": {
            "acoustic_twins": {
                "title": "Acoustic Twins (Nearest Neighbors)",
                "description": "Songs with the closest overall latent acoustic and mood profile across the entire catalog.",
                "badge": "Top Matches",
                "icon": "fa-dna",
                "songs": acoustic_twins
            },
            "same_genre": {
                "title": f"In-Genre Essentials ({seed_genre})",
                "description": f"Top related tracks sharing the exact same musical category ({seed_genre}).",
                "badge": seed_genre,
                "icon": "fa-music",
                "songs": same_genre_songs
            },
            "cross_genre": {
                "title": "Cross-Genre Bridges & Serendipity",
                "description": "Surprising songs from different genres that match this track's vibe and mood.",
                "badge": "Discovery",
                "icon": "fa-shuffle",
                "songs": cross_genre_songs
            },
            "vibe_transitions": {
                "title": "Energy & Tempo Elevation",
                "description": "Complementary tracks that maintain musical vibe while elevating energy and momentum.",
                "badge": "Energy Up",
                "icon": "fa-bolt",
                "songs": vibe_transitions
            }
        },
        "total_recommended": len(acoustic_twins) + len(same_genre_songs) + len(cross_genre_songs) + len(vibe_transitions),
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/api/users")
def get_users():
    fs: FeatureStore = STATE["feature_store"]
    users = fs.users_df.head(20).to_dict(orient="records")
    # Safely parse genre_affinities — handle both JSON strings and already-parsed dicts
    for u in users:
        ga = u.get("genre_affinities", {})
        if isinstance(ga, str):
            try:
                u["genre_affinities"] = json.loads(ga)
            except (json.JSONDecodeError, TypeError):
                u["genre_affinities"] = {}
        elif not isinstance(ga, dict):
            u["genre_affinities"] = {}
    return users


@app.post("/api/recommend")
def recommend_track(req: RecommendRequest):
    t0 = time.perf_counter()
    fs: FeatureStore = STATE["feature_store"]
    retrieval_model: TwoTowerModel = STATE["retrieval_model"]
    v_index: VectorIndex = STATE["vector_index"]
    agent: WolpertingerAgent = STATE["rl_agent"]
    env: SongRecEnv = STATE["env"]

    if req.user_id not in fs.user_id_to_idx:
        req.user_id = fs.idx_to_user_id[0]

    # 1. Candidate Generation: Two-Tower user embedding
    u_vec = fs.get_user_vector_by_id(req.user_id)
    with torch.no_grad():
        u_tensor = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0)
        u_emb = retrieval_model.user_tower(u_tensor).squeeze(0).cpu().numpy()

    top_cand_ids, cand_scores, ann_latency = v_index.query(u_emb, top_k=100)
    cand_indices = [fs.song_id_to_idx[sid] for sid in top_cand_ids]

    # Build current session state vector
    history_window = 5
    history_feats = []
    recent_sids = req.history_song_ids[-history_window:]
    recent_skips = req.history_skip_types[-history_window:]
    recent_likes = req.history_likes[-history_window:]

    pad = history_window - len(recent_sids)
    for _ in range(pad):
        history_feats.extend([0.0] * (fs.song_dim + 3))

    for idx_h, sid in enumerate(recent_sids):
        s_v = fs.get_song_vector_by_id(sid) if sid in fs.song_id_to_idx else np.zeros(fs.song_dim)
        skip_t = recent_skips[idx_h] if idx_h < len(recent_skips) else "no_skip"
        is_liked = recent_likes[idx_h] if idx_h < len(recent_likes) else 0
        resp_vec = [
            1.0 if skip_t == "no_skip" else 0.0,
            1.0 if skip_t == "skip_early" else 0.0,
            1.0 if is_liked else 0.0
        ]
        history_feats.extend(list(s_v) + resp_vec)

    ctx_vec = [14.0 / 24.0, 3.0 / 7.0, 1.0]
    total_played = max(1, len(req.history_song_ids))
    skips_cnt = sum(1 for s in req.history_skip_types if "skip" in s and s != "no_skip")
    # Clamp likes_rate to [0,1] — avoids out-of-range state features
    likes_cnt = sum(1 for lk in req.history_likes if lk == 1)
    stats_vec = [skips_cnt / total_played, min(1.0, likes_cnt / max(1, total_played)), total_played / 50.0]

    state_vec = np.concatenate([u_vec, np.array(history_feats, dtype=np.float32), np.array(ctx_vec, dtype=np.float32), np.array(stats_vec, dtype=np.float32)]).astype(np.float32)

    # 2. Re-ranking Policy
    recommendations = []
    q_score_distribution = []
    selected_proto_action = None

    if req.model_type == "random":
        chosen_indices = np.random.choice(cand_indices, size=req.slate_size, replace=False)
        for c_idx in chosen_indices:
            sid = fs.idx_to_song_id[c_idx]
            recommendations.append(fs.get_song_metadata(sid))
    elif req.model_type == "baseline":
        for i in range(req.slate_size):
            sid = top_cand_ids[i]
            recommendations.append(fs.get_song_metadata(sid))
    else:
        # Wolpertinger RL Re-ranking
        action_idx, act_info = agent.select_action(state_vec, candidate_indices=cand_indices, exploration_noise=0.0)
        selected_proto_action = act_info.get("proto_action", []).tolist() if "proto_action" in act_info else []
        
        # Format top candidate Q scores for visualizer
        sub_cands = act_info.get("candidate_indices", [])
        q_scores = act_info.get("q_scores", [])
        for c_idx, q_val in zip(sub_cands[:10], q_scores[:10]):
            c_sid = fs.idx_to_song_id[c_idx]
            c_meta = fs.get_song_metadata(c_sid)
            q_score_distribution.append({
                "song_id": c_sid,
                "title": c_meta["title"],
                "artist": c_meta["artist_name"],
                "genre": c_meta["genre"],
                "q_score": round(float(q_val), 3)
            })

        # Add top action
        top_sid = fs.idx_to_song_id[action_idx]
        recommendations.append(fs.get_song_metadata(top_sid))

        # If slate requested, take highest Q candidates
        if req.slate_size > 1:
            sorted_by_q = sorted(zip(sub_cands, q_scores), key=lambda x: x[1], reverse=True)
            for c_idx, _ in sorted_by_q[1:req.slate_size]:
                sid = fs.idx_to_song_id[c_idx]
                recommendations.append(fs.get_song_metadata(sid))

    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "user_id": req.user_id,
        "model_type": req.model_type,
        "recommendations": recommendations,
        "q_score_distribution": q_score_distribution,
        "proto_action_embedding": selected_proto_action,
        "latency_ms": round(total_latency_ms, 2)
    }


@app.post("/api/feedback")
def process_feedback(req: FeedbackRequest):
    fs: FeatureStore = STATE["feature_store"]
    bs: BehaviorStore = STATE["behavior_store"]
    song_meta = fs.get_song_metadata(req.song_id)

    # Map action_type to skip_type — disliked counts as skip_early for reward computation
    if req.action_type in ("skip_early", "disliked"):
        skip_type = "skip_early"
    elif req.action_type == "skip_late":
        skip_type = "skip_late"
    else:
        skip_type = "no_skip"

    # Compute immediate reward
    ev = PlaybackEvent(
        song_id=req.song_id,
        artist=song_meta["artist_name"],
        genre=song_meta["genre"],
        skip_type=skip_type,
        liked=1 if req.action_type == "liked" else 0,
        saved_to_playlist=1 if req.action_type in ("saved", "playlist_save") else 0
    )
    r = compute_reward(ev, [])

    # Apply additional penalty for explicit dislike beyond skip_early
    if req.action_type == "disliked":
        r -= 1.0  # Extra -1.0 on top of skip_early penalty

    # Log to BehaviorStore for hybrid recommendation signals
    bs.record_event(
        user_id=req.user_id,
        song_id=req.song_id,
        event_type=req.action_type if req.action_type != "saved" else "playlist_save",
        title=song_meta.get("title", ""),
        artist_name=song_meta.get("artist_name", ""),
        genre=song_meta.get("genre", "Pop"),
        listen_ms=req.listen_ms,
        duration_ms=song_meta.get("duration_ms", 210000),
        energy=float(song_meta.get("energy", 0.65)),
        valence=float(song_meta.get("valence", 0.60)),
        danceability=float(song_meta.get("danceability", 0.60)),
        tempo=float(song_meta.get("tempo", 120.0)),
    )

    return {
        "status": "success",
        "song_id": req.song_id,
        "action_type": req.action_type,
        "reward": round(r, 2)
    }


@app.post("/api/behavior/log")
def log_behavior_event(req: BehaviorLogRequest):
    """
    Logs a granular user behavior event to the BehaviorStore.
    Called by the frontend for: listen_end (with actual ms), replay, search_click, playlist_save.
    """
    bs: BehaviorStore = STATE["behavior_store"]
    bs.record_event(
        user_id=req.user_id,
        song_id=req.song_id,
        event_type=req.event_type,
        title=req.title,
        artist_name=req.artist_name,
        genre=req.genre,
        listen_ms=req.listen_ms,
        duration_ms=req.duration_ms,
        energy=req.energy,
        valence=req.valence,
        danceability=req.danceability,
        tempo=req.tempo,
        search_query=req.search_query,
    )
    return {"status": "ok", "user_id": req.user_id, "event_type": req.event_type}


@app.get("/api/behavior/profile/{user_id}")
def get_behavior_profile(user_id: str):
    """
    Returns a serialized snapshot of the user's real-time behavior profile.
    Used by the frontend to render User Insights panel.
    """
    bs: BehaviorStore = STATE["behavior_store"]
    profile = bs.get_full_profile_dict(user_id)

    # Add context info
    ctx_type  = infer_context_from_hour()
    ctx_info  = get_context_profile(ctx_type)
    profile["current_context"] = {"type": ctx_type, **ctx_info}
    return profile


@app.post("/api/recommend/hybrid")
def hybrid_recommend(req: HybridRecommendRequest):
    """
    Full Hybrid Recommendation Pipeline:

    1. Two-Tower candidate generation (top-100 from vector index)
    2. OpenSpot fresh candidate injection (based on liked genres/artists)
    3. Behavior signal extraction from BehaviorStore
    4. HybridRanker scoring (content + collaborative + context + diversity + exploration)
    5. Returns enriched slate with reason chips, categories, and context info
    """
    t0 = time.perf_counter()
    fs: FeatureStore = STATE["feature_store"]
    retrieval_model: TwoTowerModel = STATE["retrieval_model"]
    v_index: VectorIndex = STATE["vector_index"]
    bs: BehaviorStore = STATE["behavior_store"]
    ranker = STATE["hybrid_ranker"]

    if not fs:
        raise HTTPException(status_code=503, detail="Feature store not initialized.")

    # Fallback user
    if req.user_id not in fs.user_id_to_idx:
        req.user_id = fs.idx_to_user_id[0]

    # ── Step 1: Two-Tower candidate generation ────────────────────────────────
    u_vec = fs.get_user_vector_by_id(req.user_id)
    ann_candidates = []

    if retrieval_model and v_index:
        with torch.no_grad():
            u_tensor = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0)
            u_emb    = retrieval_model.user_tower(u_tensor).squeeze(0).cpu().numpy()
        top_ids, top_scores, _ = v_index.query(u_emb, top_k=120)
        for sid, score in zip(top_ids, top_scores):
            ann_candidates.append({"song_meta": fs.get_song_metadata(sid), "content_sim": float(score)})
    else:
        # Fallback: random sample from catalog
        sample_ids = list(fs.song_id_to_idx.keys())[:120]
        ann_candidates = [{"song_meta": fs.get_song_metadata(sid), "content_sim": 0.5} for sid in sample_ids]

    # ── Step 2: Log incoming listen_history_ms to BehaviorStore ──────────────
    for sid, ms in req.listen_history_ms.items():
        if ms > 5000 and sid in fs.song_id_to_idx:  # Only meaningful listen time (>5s)
            meta = fs.get_song_metadata(sid)
            bs.record_event(
                user_id=req.user_id, song_id=sid, event_type="listen_end",
                title=meta.get("title", ""), artist_name=meta.get("artist_name", ""),
                genre=meta.get("genre", "Pop"), listen_ms=ms,
                duration_ms=meta.get("duration_ms", 210000),
                energy=float(meta.get("energy", 0.65)),
                valence=float(meta.get("valence", 0.60)),
                danceability=float(meta.get("danceability", 0.60)),
                tempo=float(meta.get("tempo", 120.0)),
            )

    # Log replay events
    for sid in req.replay_song_ids:
        if sid in fs.song_id_to_idx:
            meta = fs.get_song_metadata(sid)
            bs.record_event(
                user_id=req.user_id, song_id=sid, event_type="replay",
                title=meta.get("title", ""), artist_name=meta.get("artist_name", ""),
                genre=meta.get("genre", "Pop"), listen_ms=0,
                duration_ms=meta.get("duration_ms", 210000),
                energy=float(meta.get("energy", 0.65)),
                valence=float(meta.get("valence", 0.60)),
                danceability=float(meta.get("danceability", 0.60)),
                tempo=float(meta.get("tempo", 120.0)),
            )

    # ── Step 3: OpenSpot fresh candidate injection ────────────────────────────
    openspot_candidates = []
    if req.pull_openspot:
        liked_genres  = bs.get_top_liked_genres(req.user_id, n=3)
        liked_artists = bs.get_top_liked_artists(req.user_id, n=3)
        ctx_type = req.context_type or infer_context_from_hour()

        # Fallback to user profile genre affinities if no behavior yet
        if not liked_genres:
            try:
                u_row = fs.users_df[fs.users_df["user_id"] == req.user_id].iloc[0]
                ga = u_row["genre_affinities"]
                if isinstance(ga, str):
                    import json as _json
                    ga = _json.loads(ga)
                liked_genres = [k for k, _ in sorted(ga.items(), key=lambda x: x[1], reverse=True)][:3]
            except Exception:
                liked_genres = ["Pop"]

        try:
            raw_tracks = openspot_client.get_recommendations_for_profile(
                liked_genres=liked_genres,
                liked_artists=liked_artists,
                context_type=ctx_type,
                limit_per_query=6,
            )
            for t in raw_tracks[:40]:
                sid = t.get("song_id", "")
                if sid and sid not in fs.song_id_to_idx:
                    # Auto-import into feature store
                    fs.add_song(t)
                    if retrieval_model and v_index:
                        try:
                            s_vec = fs.get_song_vector_by_id(sid)
                            with torch.no_grad():
                                s_t   = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
                                s_emb = retrieval_model.song_tower(s_t).squeeze(0).cpu().numpy()
                            v_index.add_item(sid, s_emb)
                        except Exception:
                            pass

                meta = fs.get_song_metadata(sid) if sid in fs.song_id_to_idx else t
                openspot_candidates.append({"song_meta": meta, "content_sim": 0.55})
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"OpenSpot profile fetch failed: {e}")

    # ── Step 4: Merge and deduplicate candidates ──────────────────────────────
    all_candidates = ann_candidates + openspot_candidates
    seen_cand_ids: set = set()
    merged_candidates = []
    for c in all_candidates:
        sid = c["song_meta"].get("song_id", "")
        if sid and sid not in seen_cand_ids:
            seen_cand_ids.add(sid)
            merged_candidates.append(c)

    # ── Step 5: Hybrid ranking ────────────────────────────────────────────────
    all_genres = sorted(set(fs.songs_df["genre"].dropna().tolist()))
    ranked = ranker.rank(
        candidates=merged_candidates,
        user_id=req.user_id,
        behavior_store=bs,
        context_type=req.context_type,
        all_genres=all_genres,
        explore_slots=req.explore_slots,
        diversity_window=req.diversity_window,
        excluded_song_ids=req.history_song_ids[-5:] if req.history_song_ids else [],
    )

    # ── Step 6: Trim to requested slate size ─────────────────────────────────
    ranked = ranked[:max(req.slate_size, 1)]

    # ── Build category summary ────────────────────────────────────────────────
    categories: Dict[str, List] = {}
    for s in ranked:
        cat = s.get("rec_category", "rl_pick")
        categories.setdefault(cat, []).append(s)

    ctx_type_final = req.context_type or infer_context_from_hour()
    ctx_info = get_context_profile(ctx_type_final)

    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "user_id":         req.user_id,
        "context_type":    ctx_type_final,
        "context_label":   ctx_info.get("label", ""),
        "context_emoji":   ctx_info.get("emoji", "🎵"),
        "recommendations": ranked,
        "categories":      {k: len(v) for k, v in categories.items()},
        "openspot_injected": len(openspot_candidates),
        "total_candidates":  len(merged_candidates),
        "latency_ms":        round(total_latency_ms, 2),
    }


@app.post("/api/simulate")
def run_simulation(req: SimulateRequest):
    """Runs a multi-session cohort simulation comparing models side-by-side."""
    env: SongRecEnv = STATE["env"]
    retrieval_model = STATE["retrieval_model"]
    vector_index = STATE["vector_index"]
    agent = STATE["rl_agent"]
    fs = STATE["feature_store"]

    results = {}
    for m_type in req.models_to_compare:
        ep_returns = []
        skip_cnt = 0
        like_cnt = 0
        tot_events = 0
        session_lens = []

        for s in range(req.num_sessions):
            obs, _ = env.reset(seed=9000 + s)
            done = False
            ep_r = 0.0
            step_c = 0

            while not done:
                u_vec = fs.get_user_vector_by_id(env.current_user_id)
                with torch.no_grad():
                    u_t = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0)
                    u_emb = retrieval_model.user_tower(u_t).squeeze(0).cpu().numpy()
                top_cand_ids, cand_scores, _ = vector_index.query(u_emb, top_k=50)
                cand_indices = [fs.song_id_to_idx[sid] for sid in top_cand_ids]

                if m_type == "random":
                    action = np.random.choice(cand_indices)
                elif m_type == "baseline":
                    cand_probs = np.exp(cand_scores[:10] / 0.1)
                    cand_probs = cand_probs / cand_probs.sum()
                    action = cand_indices[np.random.choice(len(cand_probs), p=cand_probs)]
                else:
                    action, _ = agent.select_action(obs, candidate_indices=cand_indices)

                obs, r, term, trunc, s_info = env.step(action)
                ep_r += r
                step_c += 1
                tot_events += 1
                if s_info["event"].skip_type in ["skip_early", "skip_late"]:
                    skip_cnt += 1
                if s_info["event"].liked:
                    like_cnt += 1
                if term or trunc:
                    done = True

            ep_returns.append(ep_r)
            session_lens.append(step_c)

        results[m_type] = {
            "mean_episode_return": round(float(np.mean(ep_returns)), 2),
            "skip_rate_pct": round(float(skip_cnt / max(1, tot_events)) * 100.0, 1),
            "like_rate_pct": round(float(like_cnt / max(1, tot_events)) * 100.0, 1),
            "mean_session_length": round(float(np.mean(session_lens)), 1),
            "trajectory_points": [round(float(r), 2) for r in ep_returns[:15]]
        }

    return {
        "num_sessions": req.num_sessions,
        "comparison": results
    }


@app.get("/api/metrics")
def get_offline_metrics():
    """Returns the latest offline evaluation report."""
    report_path = Path("reports/milestone4_offline_eval_report.json")
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {"message": "Report not yet generated"}


# Mount static frontend
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
