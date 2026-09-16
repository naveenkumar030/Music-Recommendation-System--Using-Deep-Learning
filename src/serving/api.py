"""
FastAPI Serving API for the RL Song Recommendation System.

Provides REST endpoints for:
- /api/health: health check and model loading status.
- /api/songs: song catalog query and audio feature lookup.
- /api/users: synthetic persona and user profile list.
- /api/recommend: next-track or 10-track slate recommendation (Two-Tower -> Wolpertinger RL -> Business Rules).
- /api/recommend/hybrid: Full hybrid pipeline (Content-Based + Collaborative + Context + Diversity + Exploration + RL Q-scores).
- /api/feedback: online feedback ingestion and behavior signal logging.
- /api/behavior/log: Granular user behavior event logging (listen_ms, replay, search_click, playlist_save).
- /api/behavior/profile/{user_id}: Serialized behavior profile with genre heatmap, acoustic centroid, mood.
- /api/openspot/import: Single-track import from OpenSpot into live RL catalog.
- /api/openspot/import/batch: Batch import of multiple OpenSpot tracks in one request.
- /api/simulate: live multi-session A/B simulation comparing models.
- /api/metrics: offline eval report and guardrails.
"""

import json
import logging
import random
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Artifact paths the server requires ────────────────────────────────────────
REQUIRED_ARTIFACTS = {
    "baseline_two_tower":   "models/baseline_two_tower.pt",
    "song_embeddings":      "models/song_embeddings.npy",
    "song_ids":             "models/song_ids.json",
    "wolpertinger_actor":   "models/wolpertinger_rl_actor.pt",
    "wolpertinger_critic":  "models/wolpertinger_rl_critic.pt",
    "user_response_model":  "models/user_response_model.joblib",
    "songs_parquet":        "data/processed/songs.parquet",
    "users_parquet":        "data/processed/users.parquet",
}

# Sidecar file for persisting OpenSpot-imported tracks across restarts
IMPORTS_SIDECAR_PATH = Path("data/processed/openspot_imports.parquet")

from src.data.feature_store import FeatureStore
from src.data.openspot_client import openspot_client
from src.data.lyrics_service import lyrics_service
from src.data.behavior_store import behavior_store, BehaviorStore
from src.env.response_model import UserResponseModel
from src.env.music_env import SongRecEnv, PlaybackEvent, compute_reward
from src.models.retrieval import TwoTowerModel, VectorIndex, UserEmbeddingCache
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
    "missing_artifacts": [],            # Tracks which artifacts failed to load
    "user_embed_cache": UserEmbeddingCache(capacity=5000),
    "all_genres_cache": None,
}


def _get_user_embedding(fs: FeatureStore, retrieval_model: TwoTowerModel, user_id: str) -> np.ndarray:
    """Retrieves user embedding from LRU cache or computes via UserSessionTower."""
    cache: Optional[UserEmbeddingCache] = STATE.get("user_embed_cache")
    if cache is not None:
        cached = cache.get(user_id)
        if cached is not None:
            return cached
    u_vec = fs.get_user_vector_by_id(user_id)
    with torch.no_grad():
        u_tensor = torch.as_tensor(u_vec, dtype=torch.float32).unsqueeze(0)
        u_emb = retrieval_model.user_tower(u_tensor).squeeze(0).cpu().numpy()
    if cache is not None:
        cache.put(user_id, u_emb)
    return u_emb


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
    """
    Initializes models and datasets into memory.
    Logs LOUD, actionable errors for every missing artifact instead of swallowing them silently.
    """
    processed_dir = "data/processed"
    models_dir = "models"
    missing: List[str] = []

    # ── Pre-flight: check every required artifact exists ─────────────────────
    for name, path in REQUIRED_ARTIFACTS.items():
        if not Path(path).exists():
            msg = (
                f"[Startup] ❌ MISSING ARTIFACT: '{path}'\n"
                f"          Run the appropriate training step to generate it:\n"
                f"            data artifacts   → python -m src.data.ingestion\n"
                f"            baseline model   → python -m src.training.train_baseline\n"
                f"            RL agent         → python -m src.training.train_rl\n"
            )
            logger.error(msg)
            missing.append(name)
    STATE["missing_artifacts"] = missing

    # ── Load FeatureStore ────────────────────────────────────────────────────
    try:
        users_df = pd.read_parquet(f"{processed_dir}/users.parquet")
        songs_df = pd.read_parquet(f"{processed_dir}/songs.parquet").replace({np.nan: None})

        # Also load any persisted OpenSpot imports from previous runs
        if IMPORTS_SIDECAR_PATH.exists():
            try:
                imports_df = pd.read_parquet(IMPORTS_SIDECAR_PATH).replace({np.nan: None})
                # Merge: only add rows with song_ids not already in songs_df
                existing_ids = set(songs_df["song_id"].tolist())
                new_rows = imports_df[~imports_df["song_id"].isin(existing_ids)]
                if len(new_rows) > 0:
                    songs_df = pd.concat([songs_df, new_rows], ignore_index=True)
                    logger.info(f"[Startup] Loaded {len(new_rows)} persisted OpenSpot imports from sidecar.")
            except Exception as sidecar_err:
                logger.warning(f"[Startup] Could not load OpenSpot imports sidecar: {sidecar_err}")

        fs = FeatureStore(processed_dir=processed_dir)
        fs.load_and_index(users_df, songs_df)
        STATE["feature_store"] = fs
        logger.info(f"[Startup] ✅ FeatureStore loaded: {fs.num_songs} songs, {fs.num_users} users.")
    except Exception as e:
        logger.error(
            f"[Startup] ❌ FATAL: Could not load FeatureStore: {e}\n"
            f"          Run: python -m src.data.ingestion"
        )
        return

    song_embeddings = None  # Track for downstream RL agent loading

    # ── Load Two-Tower Baseline ───────────────────────────────────────────────
    try:
        retrieval_model = TwoTowerModel(user_dim=fs.user_dim, song_dim=fs.song_dim, embed_dim=64)
        retrieval_model.load_state_dict(torch.load(f"{models_dir}/baseline_two_tower.pt", map_location="cpu"))
        retrieval_model.eval()
        STATE["retrieval_model"] = retrieval_model
        logger.info("[Startup] ✅ Two-Tower retrieval model loaded.")
    except Exception as e:
        logger.error(
            f"[Startup] ❌ ERROR: Could not load Two-Tower model: {e}\n"
            f"          Run: python -m src.training.train_baseline"
        )

    # ── Load Vector Index ─────────────────────────────────────────────────────
    try:
        song_embeddings = np.load(f"{models_dir}/song_embeddings.npy")
        with open(f"{models_dir}/song_ids.json") as f:
            song_ids = json.load(f)
        v_index = VectorIndex(embed_dim=64)
        v_index.build_index(song_embeddings, song_ids)
        STATE["vector_index"] = v_index
        logger.info(f"[Startup] ✅ Vector index built: {len(song_ids)} embeddings.")
    except Exception as e:
        logger.error(
            f"[Startup] ❌ ERROR: Could not load Vector Index: {e}\n"
            f"          Run: python -m src.training.train_baseline"
        )
        song_embeddings = None

    # ── Load User Response Model & Environment ────────────────────────────────
    try:
        resp_model = UserResponseModel(models_dir=models_dir)
        resp_model.load()
        STATE["response_model"] = resp_model

        env = SongRecEnv(feature_store=fs, response_model=resp_model, max_session_len=50)
        STATE["env"] = env
        logger.info(f"[Startup] ✅ SongRecEnv ready. State dim: {env.state_dim}")
    except Exception as e:
        logger.error(
            f"[Startup] ❌ ERROR: Could not load Response Model / Env: {e}\n"
            f"          Run: python -m src.training.train_rl"
        )

    # ── Load Wolpertinger RL Agent ────────────────────────────────────────────
    try:
        if song_embeddings is not None and STATE["env"] is not None:
            agent = WolpertingerAgent(
                state_dim=STATE["env"].state_dim,
                song_embeddings=song_embeddings,
                action_embed_dim=64,
                k_candidates=50
            )
            agent.load(f"{models_dir}/wolpertinger_rl")
            STATE["rl_agent"] = agent
            logger.info("[Startup] ✅ Wolpertinger RL agent loaded.")
        elif song_embeddings is None:
            logger.error(
                "[Startup] ❌ Skipping RL agent load — song_embeddings.npy missing.\n"
                "          Run: python -m src.training.train_baseline"
            )
        elif STATE["env"] is None:
            logger.error(
                "[Startup] ❌ Skipping RL agent load — SongRecEnv not initialized.\n"
                "          Run: python -m src.training.train_rl"
            )
    except Exception as e:
        logger.error(
            f"[Startup] ❌ ERROR: Could not load Wolpertinger RL agent: {e}\n"
            f"          Run: python -m src.training.train_rl"
        )


@app.get("/api/health")
def health_check():
    missing = STATE.get("missing_artifacts", [])
    models_loaded = {
        "feature_store":   STATE["feature_store"]   is not None,
        "retrieval_model": STATE["retrieval_model"] is not None,
        "vector_index":    STATE["vector_index"]    is not None,
        "rl_agent":        STATE["rl_agent"]         is not None,
        "response_model":  STATE["response_model"]  is not None,
    }
    all_loaded = all(models_loaded.values())
    return {
        "status": "healthy" if all_loaded else "degraded",
        "models_loaded": models_loaded,
        "catalog_size": STATE["feature_store"].num_songs if STATE["feature_store"] else 0,
        "missing_artifacts": missing,
        "fix_commands": {
            "data":     "python -m src.data.ingestion",
            "baseline": "python -m src.training.train_baseline",
            "rl_agent": "python -m src.training.train_rl",
        } if missing else {},
        "active_pipeline": "wolpertinger_rl" if models_loaded["rl_agent"] else "hybrid_heuristic",
    }


@app.get("/api/songs")
def get_songs(
    genre: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Returns paginated song catalog with total count.
    - search queries: uncapped (up to 500 results) so search-to-select covers the full catalog.
    - browse (no search): paginated with offset/limit.
    Response: {"songs": [...], "total": N, "offset": offset, "limit": limit}
    """
    fs: FeatureStore = STATE["feature_store"]
    df = fs.songs_df.copy()
    if genre and genre != "All":
        df = df[df["genre"] == genre]
    if search:
        s_lower = search.lower()
        mask = (
            df["title"].str.lower().str.contains(s_lower, na=False) |
            df["artist_name"].str.lower().str.contains(s_lower, na=False) |
            df["song_id"].str.lower().str.contains(s_lower, na=False)
        )
        df = df[mask]
        # For search-to-select, return all matches (capped at 500) — no pagination offset
        total = len(df)
        records = df.head(500).replace({np.nan: None}).to_dict(orient="records")
        return {"songs": records, "total": total, "offset": 0, "limit": 500}
    # Browse mode — paginate with offset
    total = len(df)
    page_df = df.iloc[offset: offset + limit]
    records = page_df.replace({np.nan: None}).to_dict(orient="records")
    return {"songs": records, "total": total, "offset": offset, "limit": limit}


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
    Also persists the track to disk so it survives server restarts.
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

    # 5. Persist to sidecar parquet so the track survives restarts
    _persist_import_to_sidecar(track_dict)

    return {
        "status": "success",
        "message": f"Successfully imported '{req.title}' by {req.artist_name} into RL catalog!",
        "song": fs.get_song_metadata(song_id),
        "catalog_size": fs.num_songs
    }


class OpenSpotBatchImportRequest(BaseModel):
    """Request body for bulk-importing multiple OpenSpot tracks in one call."""
    tracks: List[OpenSpotImportRequest]


@app.post("/api/openspot/import/batch")
def openspot_import_batch(req: OpenSpotBatchImportRequest):
    """
    Batch-imports a list of OpenSpot tracks into the live RL catalog.
    One bad track doesn't fail the whole batch — errors are reported per-track.
    All successfully imported tracks are also persisted to disk.
    """
    fs: FeatureStore = STATE["feature_store"]
    retrieval_model: TwoTowerModel = STATE["retrieval_model"]
    v_index: VectorIndex = STATE["vector_index"]
    agent: WolpertingerAgent = STATE["rl_agent"]

    if not fs or not retrieval_model or not v_index:
        raise HTTPException(status_code=503, detail="Server models not initialized.")

    imported = []
    failed   = []
    tracks_to_persist = []

    for t in req.tracks:
        try:
            track_dict = t.dict()
            song_id    = t.song_id

            # 1. Index into FeatureStore
            fs.add_song(track_dict)

            # 2. Encode with Song Tower
            s_vec = fs.get_song_vector_by_id(song_id)
            with torch.no_grad():
                s_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
                s_emb    = retrieval_model.song_tower(s_tensor).squeeze(0).cpu().numpy()

            # 3. Add to Vector Index
            v_index.add_item(song_id, s_emb)

            # 4. Update RL agent candidate space
            if agent is not None:
                agent.add_song_embedding(s_emb)

            imported.append(song_id)
            tracks_to_persist.append(track_dict)
            logger.info(f"[BatchImport] ✅ Imported '{t.title}' by {t.artist_name} ({song_id})")
        except Exception as e:
            logger.warning(f"[BatchImport] ❌ Failed to import track '{t.song_id}': {e}")
            failed.append({"song_id": t.song_id, "title": t.title, "error": str(e)})

    # 5. Persist all successful imports to sidecar
    if tracks_to_persist:
        _persist_import_to_sidecar(tracks_to_persist)

    return {
        "status":       "success" if not failed else "partial",
        "imported":     imported,
        "failed":       failed,
        "catalog_size": fs.num_songs,
        "message":      f"Imported {len(imported)} tracks; {len(failed)} failed.",
    }


def _persist_import_to_sidecar(track_or_tracks):
    """
    Appends imported track(s) to the OpenSpot imports sidecar parquet file.
    This ensures imported tracks survive server restarts.
    """
    try:
        if isinstance(track_or_tracks, dict):
            rows = [track_or_tracks]
        else:
            rows = list(track_or_tracks)

        new_df = pd.DataFrame(rows)

        if IMPORTS_SIDECAR_PATH.exists():
            existing = pd.read_parquet(IMPORTS_SIDECAR_PATH)
            existing_ids = set(existing["song_id"].tolist())
            new_rows = new_df[~new_df["song_id"].isin(existing_ids)]
            if len(new_rows) > 0:
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined.to_parquet(IMPORTS_SIDECAR_PATH, index=False)
                logger.info(f"[Persist] Appended {len(new_rows)} new tracks to sidecar.")
        else:
            IMPORTS_SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_parquet(IMPORTS_SIDECAR_PATH, index=False)
            logger.info(f"[Persist] Created sidecar with {len(new_df)} imported tracks.")
    except Exception as e:
        logger.warning(f"[Persist] Could not persist OpenSpot imports to disk: {e}")


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


def _build_state_vector(
    fs: FeatureStore,
    user_id: str,
    history_sids: List[str],
    history_skips: List[str],
    history_likes: List[int],
    history_window: int = 5
) -> np.ndarray:
    """Constructs flat observation vector matching MDP state specification."""
    if not fs or user_id not in fs.user_id_to_idx:
        u_vec = np.zeros(fs.user_dim if fs else 13, dtype=np.float32)
    else:
        u_vec = fs.get_user_vector_by_id(user_id)

    history_feats = []
    recent_sids = history_sids[-history_window:]
    recent_skips = history_skips[-history_window:]
    recent_likes = history_likes[-history_window:]

    pad = history_window - len(recent_sids)
    for _ in range(pad):
        history_feats.extend([0.0] * (fs.song_dim + 3 if fs else 16))

    for idx_h, sid in enumerate(recent_sids):
        s_v = fs.get_song_vector_by_id(sid) if (fs and sid in fs.song_id_to_idx) else np.zeros(fs.song_dim if fs else 13)
        skip_t = recent_skips[idx_h] if idx_h < len(recent_skips) else "no_skip"
        is_liked = recent_likes[idx_h] if idx_h < len(recent_likes) else 0
        resp_vec = [
            1.0 if skip_t == "no_skip" else 0.0,
            1.0 if skip_t == "skip_early" else 0.0,
            1.0 if is_liked else 0.0
        ]
        history_feats.extend(list(s_v) + resp_vec)

    ctx_vec = [14.0 / 24.0, 3.0 / 7.0, 1.0]
    total_played = max(1, len(history_sids))
    skips_cnt = sum(1 for s in history_skips if "skip" in s and s != "no_skip")
    likes_cnt = sum(1 for lk in history_likes if lk == 1)
    stats_vec = [skips_cnt / total_played, likes_cnt / 10.0, min(1.0, total_played / 50.0)]

    state_vec = np.concatenate([
        u_vec,
        np.array(history_feats, dtype=np.float32),
        np.array(ctx_vec, dtype=np.float32),
        np.array(stats_vec, dtype=np.float32)
    ]).astype(np.float32)
    return state_vec


@app.post("/api/recommend")
def recommend_track(req: RecommendRequest):
    t0 = time.perf_counter()
    fs: FeatureStore = STATE["feature_store"]
    retrieval_model: TwoTowerModel = STATE["retrieval_model"]
    v_index: VectorIndex = STATE["vector_index"]
    agent: WolpertingerAgent = STATE["rl_agent"]
    env: SongRecEnv = STATE["env"]

    if not fs:
        raise HTTPException(status_code=503, detail="Feature store not initialized.")

    if req.user_id not in fs.user_id_to_idx:
        req.user_id = fs.idx_to_user_id[0]

    # 1. Candidate Generation: Cached Two-Tower user embedding
    u_emb = _get_user_embedding(fs, retrieval_model, req.user_id)
    top_cand_ids, cand_scores, ann_latency = v_index.query(u_emb, top_k=100)
    cand_indices = [fs.song_id_to_idx[sid] for sid in top_cand_ids]

    # Build current session state vector
    state_vec = _build_state_vector(
        fs=fs,
        user_id=req.user_id,
        history_sids=req.history_song_ids,
        history_skips=req.history_skip_types,
        history_likes=req.history_likes
    )

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
        # Cold-start exploration: when session history is empty, inject noise so different
        # users get varied first picks instead of converging to the same zero-padded state.
        cold_start_noise = 0.15 if len(req.history_song_ids) == 0 else 0.0
        recent_actions = [fs.song_id_to_idx[sid] for sid in req.history_song_ids if sid in fs.song_id_to_idx]
        action_idx, act_info = agent.select_action(
            state_vec,
            candidate_indices=cand_indices,
            exploration_noise=cold_start_noise,
            recent_action_indices=recent_actions
        )
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
    agent: Optional[WolpertingerAgent] = STATE.get("rl_agent")

    if not fs:
        raise HTTPException(status_code=503, detail="FeatureStore not initialized.")

    song_meta = fs.get_song_metadata(req.song_id) if req.song_id in fs.song_id_to_idx else {
        "title": "Unknown Song", "artist_name": "Unknown Artist", "genre": "Pop",
        "duration_ms": 210000, "energy": 0.65, "valence": 0.60, "danceability": 0.60, "tempo": 120.0
    }

    # Map action_type to skip_type — disliked counts as skip_early for reward computation
    if req.action_type in ("skip_early", "disliked"):
        skip_type = "skip_early"
    elif req.action_type == "skip_late":
        skip_type = "skip_late"
    else:
        skip_type = "no_skip"

    # Maintain active session transitions for online RL learning
    session_key = req.session_id or req.user_id or "default_session"
    if "active_sessions" not in STATE or not isinstance(STATE["active_sessions"], dict):
        STATE["active_sessions"] = {}

    session_data = STATE["active_sessions"].setdefault(session_key, {
        "history_sids": [],
        "history_skips": [],
        "history_likes": [],
        "recent_events": []
    })

    # State before this interaction s_t
    s_t = _build_state_vector(
        fs, req.user_id,
        session_data["history_sids"],
        session_data["history_skips"],
        session_data["history_likes"]
    )

    # Compute immediate reward using RL specification:
    # Like: +1.0, Listen (no_skip): +0.5, Skip: -1.0
    ev = PlaybackEvent(
        song_id=req.song_id,
        artist=song_meta.get("artist_name", "Unknown Artist"),
        genre=song_meta.get("genre", "Pop"),
        skip_type=skip_type,
        liked=1 if req.action_type == "liked" else 0,
        saved_to_playlist=1 if req.action_type in ("saved", "playlist_save") else 0
    )
    r = compute_reward(ev, session_data["recent_events"])

    if req.action_type == "disliked":
        r -= 0.5  # Extra penalty for explicit dislike

    # Update session history
    session_data["history_sids"].append(req.song_id)
    session_data["history_skips"].append(skip_type)
    session_data["history_likes"].append(1 if req.action_type == "liked" else 0)
    session_data["recent_events"].append(ev)
    if len(session_data["recent_events"]) > 25:
        session_data["recent_events"] = session_data["recent_events"][-25:]

    # State after interaction s_{t+1}
    s_next = _build_state_vector(
        fs, req.user_id,
        session_data["history_sids"],
        session_data["history_skips"],
        session_data["history_likes"]
    )

    action_idx = fs.song_id_to_idx.get(req.song_id, 0)
    loss_info = {}
    policy_updated = False

    # Push to Replay Buffer & trigger online policy update step
    if agent is not None:
        try:
            agent.replay_buffer.push(s_t, action_idx, float(r), s_next, False)
            if len(agent.replay_buffer) >= 4:
                loss_info = agent.train_step(batch_size=32)
                policy_updated = bool(loss_info)
        except Exception as err:
            logger.warning(f"[RL Online Learning] Online train step warning: {err}")

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

    # Dynamic User State snapshot
    profile = bs.get_full_profile_dict(req.user_id)
    recent_sids = session_data["history_sids"][-5:]
    recent_songs_meta = [fs.get_song_metadata(sid) for sid in recent_sids if sid in fs.song_id_to_idx]
    recent_artists = [m.get("artist_name", "") for m in recent_songs_meta if m.get("artist_name")]

    user_state = {
        "user_id": req.user_id,
        "genre": profile.get("top_genre") or song_meta.get("genre", "Pop"),
        "genre_affinities": profile.get("genre_affinities", {}),
        "artist": song_meta.get("artist_name", "Unknown Artist"),
        "recent_artists": recent_artists,
        "mood": profile.get("acoustic_centroid", {"energy": 0.65, "valence": 0.60}),
        "recent_songs": recent_songs_meta,
        "total_interactions": len(session_data["history_sids"])
    }

    return {
        "status": "success",
        "song_id": req.song_id,
        "action_type": req.action_type,
        "reward": round(r, 2),
        "policy_updated": policy_updated,
        "loss_info": loss_info,
        "buffer_size": len(agent.replay_buffer) if agent else 0,
        "user_state": user_state
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
        u_emb = _get_user_embedding(fs, retrieval_model, req.user_id)
        top_ids, top_scores, _ = v_index.query(u_emb, top_k=120)
        for sid, score in zip(top_ids, top_scores):
            ann_candidates.append({"song_meta": fs.get_song_metadata(sid), "content_sim": float(score)})
    else:
        # Fallback: random sample weighted by popularity (avoids insertion-order bias)
        all_ids = list(fs.song_id_to_idx.keys())
        # Try to sort by popularity descending; fall back to random if field missing
        try:
            all_ids.sort(
                key=lambda sid: float(fs.get_song_metadata(sid).get("popularity", 50) or 50),
                reverse=True
            )
            sample_ids = all_ids[:120]
        except Exception:
            sample_ids = random.sample(all_ids, min(120, len(all_ids)))
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
                liked_genres=liked_genres[:1],
                liked_artists=liked_artists[:1],
                context_type=ctx_type,
                limit_per_query=4,
            )
            for t in raw_tracks[:15]:
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

    # ── Step 5: Cold-start prior — seed genre affinity from user profile ─────
    # When behavior_store has no events yet, genre/artist weights are all 0.
    # Seed them from the persisted user profile so the ranker has a prior.
    profile_prior: Dict[str, float] = {}
    try:
        genre_weights_live = bs.get_recency_weighted_genre_preferences(req.user_id)
        if not genre_weights_live:  # Empty = no behavior events yet
            u_row = fs.users_df[fs.users_df["user_id"] == req.user_id]
            if len(u_row) > 0:
                ga = u_row.iloc[0]["genre_affinities"]
                if isinstance(ga, str):
                    import json as _json
                    ga = _json.loads(ga)
                if isinstance(ga, dict):
                    profile_prior = ga  # Will be injected into ranker below
    except Exception:
        pass

    # ── Step 6: Hybrid ranking ────────────────────────────────────────────────
    all_genres = STATE.get("all_genres_cache")
    if not all_genres:
        all_genres = sorted(set(fs.songs_df["genre"].dropna().tolist()))
        STATE["all_genres_cache"] = all_genres

    # Hard-exclude all songs the user skip_early'd in this session (not just last 5)
    hard_excluded = set(req.history_song_ids[-5:] if req.history_song_ids else [])
    if req.history_song_ids and req.history_skip_types:
        for sid, stype in zip(req.history_song_ids, req.history_skip_types):
            if stype in ("skip_early", "disliked"):
                hard_excluded.add(sid)

    ranked = ranker.rank(
        candidates=merged_candidates,
        user_id=req.user_id,
        behavior_store=bs,
        context_type=req.context_type,
        all_genres=all_genres,
        explore_slots=req.explore_slots,
        diversity_window=req.diversity_window,
        excluded_song_ids=list(hard_excluded),
        profile_genre_prior=profile_prior,
    )

    # ── Step 7: Trim to requested slate size ──────────────────────────────────
    ranked = ranked[:max(req.slate_size, 1)]

    # ── Step 8: Augment with Wolpertinger Q-scores when RL agent is available ─
    agent: WolpertingerAgent = STATE.get("rl_agent")
    q_score_distribution = []
    if agent is not None and STATE.get("env") is not None:
        try:
            # Build a minimal state vector for RL scoring
            env = STATE["env"]
            history_window = 5
            history_feats = []
            recent_sids   = req.history_song_ids[-history_window:]
            recent_skips  = req.history_skip_types[-history_window:]
            recent_likes  = req.history_likes[-history_window:]
            pad = history_window - len(recent_sids)
            for _ in range(pad):
                history_feats.extend([0.0] * (fs.song_dim + 3))
            for idx_h, sid in enumerate(recent_sids):
                s_v    = fs.get_song_vector_by_id(sid) if sid in fs.song_id_to_idx else np.zeros(fs.song_dim)
                skip_t = recent_skips[idx_h] if idx_h < len(recent_skips) else "no_skip"
                is_lk  = recent_likes[idx_h] if idx_h < len(recent_likes) else 0
                history_feats.extend(list(s_v) + [
                    1.0 if skip_t == "no_skip" else 0.0,
                    1.0 if skip_t == "skip_early" else 0.0,
                    1.0 if is_lk else 0.0,
                ])
            ctx_vec   = [14.0 / 24.0, 3.0 / 7.0, 1.0]
            total_played = max(1, len(req.history_song_ids))
            skips_cnt    = sum(1 for s in req.history_skip_types if "skip" in s and s != "no_skip")
            likes_cnt    = sum(1 for lk in req.history_likes if lk == 1)
            stats_vec    = [skips_cnt / total_played, likes_cnt / 10.0, total_played / 50.0]
            state_vec    = np.concatenate([
                fs.get_user_vector_by_id(req.user_id),
                np.array(history_feats, dtype=np.float32),
                np.array(ctx_vec, dtype=np.float32),
                np.array(stats_vec, dtype=np.float32),
            ]).astype(np.float32)

            # Score top candidates from the ranked slate
            cand_ids = [s["song_id"] for s in ranked if s.get("song_id") in fs.song_id_to_idx]
            cand_indices = [fs.song_id_to_idx[sid] for sid in cand_ids]
            if cand_indices:
                _, act_info = agent.select_action(state_vec, candidate_indices=cand_indices, exploration_noise=0.0)
                sub_cands = act_info.get("candidate_indices", [])
                q_scores  = act_info.get("q_scores", [])
                for c_idx, q_val in zip(sub_cands[:len(ranked)], q_scores[:len(ranked)]):
                    c_sid  = fs.idx_to_song_id.get(c_idx, "")
                    c_meta = fs.get_song_metadata(c_sid) if c_sid else {}
                    q_score_distribution.append({
                        "song_id": c_sid,
                        "title":   c_meta.get("title", ""),
                        "artist":  c_meta.get("artist_name", ""),
                        "genre":   c_meta.get("genre", ""),
                        "q_score": round(float(q_val), 3),
                    })
        except Exception as rl_err:
            logger.warning(f"[Hybrid] RL Q-score augmentation failed (non-fatal): {rl_err}")

    # ── Build category summary ────────────────────────────────────────────────
    categories: Dict[str, List] = {}
    for s in ranked:
        cat = s.get("rec_category", "rl_pick")
        categories.setdefault(cat, []).append(s)

    ctx_type_final = req.context_type or infer_context_from_hour()
    ctx_info = get_context_profile(ctx_type_final)

    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "user_id":             req.user_id,
        "context_type":        ctx_type_final,
        "context_label":       ctx_info.get("label", ""),
        "context_emoji":       ctx_info.get("emoji", "🎵"),
        "recommendations":     ranked,
        "q_score_distribution": q_score_distribution,
        "categories":          {k: len(v) for k, v in categories.items()},
        "openspot_injected":   len(openspot_candidates),
        "total_candidates":    len(merged_candidates),
        "active_pipeline":     "wolpertinger_rl" if agent is not None else "hybrid_heuristic",
        "latency_ms":          round(total_latency_ms, 2),
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
            session_actions = []

            while not done:
                u_emb = _get_user_embedding(fs, retrieval_model, env.current_user_id)
                top_cand_ids, cand_scores, _ = vector_index.query(u_emb, top_k=50)
                cand_indices = [fs.song_id_to_idx[sid] for sid in top_cand_ids]

                if m_type == "random":
                    action = np.random.choice(cand_indices)
                elif m_type == "baseline":
                    cand_probs = np.exp(cand_scores[:10] / 0.1)
                    cand_probs = cand_probs / cand_probs.sum()
                    action = cand_indices[np.random.choice(len(cand_probs), p=cand_probs)]
                else:
                    action, _ = agent.select_action(
                        obs,
                        candidate_indices=cand_indices,
                        recent_action_indices=session_actions
                    )

                session_actions.append(action)
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
