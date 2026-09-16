"""
HybridRanker — Weighted Scoring Hybrid Recommendation Engine.

Combines:
- Content-Based Filtering (Two-Tower cosine similarity)
- Collaborative signals (recency-weighted likes/skips/replays from BehaviorStore)
- Contextual features (time-of-day, mood, activity context)
- Diversity penalty (consecutive same-genre penalty)
- Exploration bonus (unfamiliar genres get a boost)

Operates in two modes:
1. Weighted scoring (always available — no training needed)
2. LightGBM LambdaRank (optional — falls back to weighted scoring if lgbm not installed)

All scoring is done at inference time; no offline training required for the weighted mode.
"""

import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ─── Context Inference ────────────────────────────────────────────────────────

CONTEXT_PROFILES = {
    "morning_focus": {
        "pref_energy": 0.55, "pref_valence": 0.65, "pref_instrumentalness": 0.35,
        "label": "☕ Morning Focus", "emoji": "☕",
        "description": "Calm, focused music to start the day"
    },
    "afternoon_energy": {
        "pref_energy": 0.80, "pref_valence": 0.70, "pref_instrumentalness": 0.05,
        "label": "⚡ Afternoon Energy", "emoji": "⚡",
        "description": "High-energy tracks to power through the afternoon"
    },
    "evening_chill": {
        "pref_energy": 0.42, "pref_valence": 0.55, "pref_instrumentalness": 0.20,
        "label": "🌆 Evening Chill", "emoji": "🌆",
        "description": "Relaxed vibes for winding down"
    },
    "night_study": {
        "pref_energy": 0.30, "pref_valence": 0.45, "pref_instrumentalness": 0.65,
        "label": "🌙 Night Study", "emoji": "🌙",
        "description": "Lo-fi and instrumental for late-night focus"
    },
    "workout": {
        "pref_energy": 0.92, "pref_valence": 0.72, "pref_instrumentalness": 0.02,
        "label": "🏋️ Workout", "emoji": "🏋️",
        "description": "High BPM bangers for maximum performance"
    },
    "travel": {
        "pref_energy": 0.62, "pref_valence": 0.68, "pref_instrumentalness": 0.10,
        "label": "✈️ Travel", "emoji": "✈️",
        "description": "Feel-good road trip anthems"
    },
}


def infer_context_from_hour(hour: Optional[int] = None) -> str:
    """
    Infers listening context from time of day.
    Falls back to current local hour if not provided.
    """
    if hour is None:
        hour = datetime.now().hour

    if 5 <= hour < 10:
        return "morning_focus"
    elif 10 <= hour < 17:
        return "afternoon_energy"
    elif 17 <= hour < 21:
        return "evening_chill"
    else:
        return "night_study"


def get_context_profile(context_type: str) -> Dict[str, Any]:
    """Returns the acoustic target profile for a context type."""
    return CONTEXT_PROFILES.get(context_type, CONTEXT_PROFILES["afternoon_energy"])


# ─── Feature Extraction ───────────────────────────────────────────────────────

def build_candidate_features(
    song_meta: Dict,
    content_sim_score: float,
    user_genre_weights: Dict[str, float],
    user_artist_weights: Dict[str, float],
    acoustic_centroid: Dict[str, float],
    context_type: str,
    consecutive_genre_count: int,
    genres_heard: set,
    replay_song_ids: List[str],
    liked_song_ids: List[str],
    skipped_song_ids: List[str],
    all_genres: List[str],
    profile_genre_prior: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Builds the full feature vector for one (user, candidate song) pair.
    Returns a dict of named scores (useful for explainability).

    When user_genre_weights is empty (cold start), blends in profile_genre_prior
    so the ranker has a meaningful prior instead of scoring every genre at 0.
    """
    genre = song_meta.get("genre", "Pop")
    artist = song_meta.get("artist_name", "")
    song_id = song_meta.get("song_id", "")

    # ── Content similarity (Two-Tower) ────────────────────────────────────────
    f_content = float(np.clip(content_sim_score, -1.0, 1.0))

    # ── Genre affinity (collaborative signal) ─────────────────────────────────
    # Cold-start: if live weights are empty, blend in the persisted profile prior
    effective_genre_weights = user_genre_weights
    if not user_genre_weights and profile_genre_prior:
        effective_genre_weights = profile_genre_prior  # raw profile affinities (0–1 scale)
    f_genre_affinity = float(effective_genre_weights.get(genre, 0.0))

    # ── Artist affinity (collaborative signal) ────────────────────────────────
    f_artist_affinity = user_artist_weights.get(artist, 0.0)

    # ── Acoustic delta from user preference centroid ──────────────────────────
    e_diff = abs(song_meta.get("energy", 0.65) - acoustic_centroid.get("energy", 0.65))
    v_diff = abs(song_meta.get("valence", 0.60) - acoustic_centroid.get("valence", 0.60))
    d_diff = abs(song_meta.get("danceability", 0.60) - acoustic_centroid.get("danceability", 0.60))
    f_acoustic_match = 1.0 - float(np.clip((e_diff + v_diff + d_diff) / 3.0, 0.0, 1.0))

    # ── Context boost ─────────────────────────────────────────────────────────
    ctx = get_context_profile(context_type)
    ctx_e_diff = abs(song_meta.get("energy", 0.65) - ctx["pref_energy"])
    ctx_v_diff = abs(song_meta.get("valence", 0.60) - ctx["pref_valence"])
    ctx_i_diff = abs(song_meta.get("instrumentalness", 0.0) - ctx["pref_instrumentalness"])
    f_context   = 1.0 - float(np.clip((ctx_e_diff + ctx_v_diff + ctx_i_diff) / 3.0, 0.0, 1.0))

    # ── Popularity ────────────────────────────────────────────────────────────
    f_popularity = float(np.clip(song_meta.get("popularity", 50) / 100.0, 0.0, 1.0))

    # ── Diversity penalty ─────────────────────────────────────────────────────
    # Penalize if consecutive same-genre count is high (> 2)
    f_diversity_penalty = 0.0
    if consecutive_genre_count >= 3:
        f_diversity_penalty = min(0.6, 0.15 * (consecutive_genre_count - 2))

    # ── Exploration bonus ─────────────────────────────────────────────────────
    # Bonus for genres not yet heard in the session
    f_exploration = 0.0
    if genre not in genres_heard and len(genres_heard) > 2:
        f_exploration = 0.25  # Reward novel genre discovery

    # ── Replay & Like recency bonus ───────────────────────────────────────────
    f_replay_bonus = 0.30 if song_id in replay_song_ids else 0.0
    f_liked_bonus  = 0.40 if song_id in liked_song_ids  else 0.0

    # ── Skip penalty ──────────────────────────────────────────────────────────
    f_skip_penalty = 0.80 if song_id in skipped_song_ids else 0.0

    return {
        "content_sim":        f_content,
        "genre_affinity":     f_genre_affinity,
        "artist_affinity":    f_artist_affinity,
        "acoustic_match":     f_acoustic_match,
        "context_boost":      f_context,
        "popularity":         f_popularity,
        "diversity_penalty":  f_diversity_penalty,
        "exploration_bonus":  f_exploration,
        "replay_bonus":       f_replay_bonus,
        "liked_bonus":        f_liked_bonus,
        "skip_penalty":       f_skip_penalty,
    }


# ─── Weighted Scoring ─────────────────────────────────────────────────────────

# Weight vector — tuned for optimal balance of content relevance, user affinity, and calibrated exploration
WEIGHTS = {
    "content_sim":       0.35,
    "genre_affinity":    0.25,
    "artist_affinity":   0.15,
    "acoustic_match":    0.15,
    "context_boost":     0.10,
    "popularity":        0.05,
    "diversity_penalty": -0.30,   # Scaled penalty for repeats
    "exploration_bonus": 0.20,    # Controlled discovery boost
    "replay_bonus":      0.35,    # Strong affinity for replayed tracks
    "liked_bonus":       0.40,    # High priority for liked-similar tracks
    "skip_penalty":      -0.80,   # Strong deduction for skipped tracks
}


def weighted_score(features: Dict[str, float]) -> float:
    """Computes a scalar recommendation score from feature dict using fixed weights."""
    score = 0.0
    for feat, w in WEIGHTS.items():
        score += features.get(feat, 0.0) * w
    return float(score)


# ─── Recommendation Reason Explainer ─────────────────────────────────────────

def get_recommendation_reason(features: Dict[str, float], context_type: str, song_meta: Dict) -> Dict[str, str]:
    """
    Returns a human-readable reason label + icon for why this song was recommended.
    """
    ctx_label = get_context_profile(context_type).get("label", "")

    if features.get("liked_bonus", 0) > 0:
        return {"icon": "fa-heart", "label": "Based on your likes", "category": "because_you_liked", "color": "#ff6b6b"}
    if features.get("replay_bonus", 0) > 0:
        return {"icon": "fa-rotate-right", "label": "You replayed similar", "category": "more_like_replays", "color": "#f7b731"}
    if features.get("exploration_bonus", 0) > 0:
        return {"icon": "fa-compass", "label": "Explore New", "category": "explore_new", "color": "#45aaf2"}
    if features.get("artist_affinity", 0) > 0.15:
        return {"icon": "fa-microphone", "label": "Liked artist", "category": "because_you_liked", "color": "#ff6b6b"}
    if features.get("genre_affinity", 0) > 0.15:
        return {"icon": "fa-music", "label": f"Fav genre match", "category": "because_you_liked", "color": "#ff6b6b"}
    if features.get("context_boost", 0) > 0.70:
        return {"icon": "fa-clock", "label": ctx_label, "category": "right_now", "color": "#a55eea"}
    if features.get("content_sim", 0) > 0.60:
        return {"icon": "fa-dna", "label": "Acoustic twin", "category": "because_you_liked", "color": "#26de81"}
    if song_meta.get("audio_url"):
        return {"icon": "fa-fire", "label": "Trending OpenSpot", "category": "trending_openspot", "color": "#fd9644"}
    return {"icon": "fa-brain", "label": "RL Pick", "category": "rl_pick", "color": "#778ca3"}


# ─── Main Ranker Class ────────────────────────────────────────────────────────

class HybridRanker:
    """
    Re-ranks candidate songs using a combination of content, collaborative,
    contextual, diversity, and exploration signals.

    Usage:
        ranker = HybridRanker()
        ranked = ranker.rank(
            candidates=[{"song_meta": {...}, "content_sim": 0.82}, ...],
            user_id="usr_001",
            behavior_store=behavior_store,
            context_type="evening_chill",
            all_genres=[...],
            explore_slots=2,
            diversity_window=3,
        )
    """

    def __init__(self):
        self._lgbm_model = None
        self._try_load_lgbm()

    def _try_load_lgbm(self):
        """Tries to load a pre-trained LightGBM ranker model."""
        try:
            import lightgbm as lgb
            import os
            model_path = "models/hybrid_ranker.lgbm"
            if os.path.exists(model_path):
                self._lgbm_model = lgb.Booster(model_file=model_path)
                logger.info("[HybridRanker] LightGBM ranker model loaded.")
            else:
                logger.info("[HybridRanker] No LightGBM model found — using weighted scoring.")
        except ImportError:
            logger.info("[HybridRanker] LightGBM not installed — using weighted scoring.")

    def rank(
        self,
        candidates: List[Dict],   # Each: {"song_meta": {...}, "content_sim": float}
        user_id: str,
        behavior_store: Any,
        context_type: Optional[str] = None,
        all_genres: Optional[List[str]] = None,
        explore_slots: int = 2,
        diversity_window: int = 3,
        excluded_song_ids: Optional[List[str]] = None,
        profile_genre_prior: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Ranks candidate songs and returns an enriched list with scores and reasons.

        Args:
            candidates: List of dicts with keys 'song_meta' and 'content_sim'
            user_id: The current user
            behavior_store: BehaviorStore instance
            context_type: Context string (e.g. 'evening_chill'); auto-inferred if None
            all_genres: Full genre list for exploration tracking
            explore_slots: How many exploration picks to force into the final slate
            diversity_window: Number of recent picks to track for same-genre penalty
            excluded_song_ids: Song IDs to exclude (e.g. currently playing)
            profile_genre_prior: Persisted genre affinity dict from user profile.
                Used as cold-start prior when behavior_store has no events yet.

        Returns:
            Ranked list of enriched song dicts with 'rec_score', 'rec_reason', 'rec_category'
        """
        t0 = time.perf_counter()
        excluded = set(excluded_song_ids or [])

        # Auto-infer context if not provided
        if not context_type:
            context_type = infer_context_from_hour()

        # Pull behavior signals
        genre_weights    = behavior_store.get_recency_weighted_genre_preferences(user_id)
        artist_weights   = behavior_store.get_recency_weighted_artist_preferences(user_id)
        acoustic_centroid= behavior_store.get_preferred_acoustic_centroid(user_id)
        replay_ids       = behavior_store.get_replay_song_ids(user_id)
        profile          = behavior_store.get_profile(user_id)
        liked_ids        = profile.liked_song_ids[:50]
        skipped_ids      = set(profile.skipped_song_ids[-50:])
        recent_ids       = profile.recent_song_ids[:30]

        all_genres = all_genres or []
        genres_heard = {e.genre for e in list(behavior_store._events.get(user_id, []))}

        # Score all candidates
        scored = []
        for cand in candidates:
            meta = cand.get("song_meta", {})
            sid  = meta.get("song_id", "")

            if sid in excluded:
                continue

            features = build_candidate_features(
                song_meta=meta,
                content_sim_score=cand.get("content_sim", 0.5),
                user_genre_weights=genre_weights,
                user_artist_weights=artist_weights,
                acoustic_centroid=acoustic_centroid,
                context_type=context_type,
                consecutive_genre_count=0,  # computed after sorting
                genres_heard=genres_heard,
                replay_song_ids=replay_ids,
                liked_song_ids=liked_ids,
                skipped_song_ids=list(skipped_ids),
                all_genres=all_genres,
                profile_genre_prior=profile_genre_prior,
            )

            if self._lgbm_model is not None:
                score = self._lgbm_score(features)
            else:
                score = weighted_score(features)

            reason = get_recommendation_reason(features, context_type, meta)

            scored.append({
                **meta,
                "rec_score": round(score, 4),
                "rec_features": features,
                "rec_reason": reason["label"],
                "rec_reason_icon": reason["icon"],
                "rec_reason_color": reason["color"],
                "rec_category": reason["category"],
                "context_type": context_type,
                "context_label": get_context_profile(context_type).get("label", ""),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["rec_score"], reverse=True)

        # ── Diversity post-processing ─────────────────────────────────────────
        # Interleave results to prevent same-genre runs
        final_ranked = self._apply_diversity(scored, diversity_window=diversity_window)

        # ── Inject exploration slots ──────────────────────────────────────────
        final_ranked = self._inject_exploration(
            final_ranked, scored, genres_heard, all_genres, explore_slots
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"[HybridRanker] Ranked {len(final_ranked)} songs in {latency_ms:.1f}ms for user {user_id}")

        return final_ranked

    def _lgbm_score(self, features: Dict[str, float]) -> float:
        """Scores a single candidate using the LightGBM model."""
        try:
            feat_vec = np.array([[
                features["content_sim"],
                features["genre_affinity"],
                features["artist_affinity"],
                features["acoustic_match"],
                features["context_boost"],
                features["popularity"],
                features["replay_bonus"],
                features["liked_bonus"],
            ]], dtype=np.float32)
            return float(self._lgbm_model.predict(feat_vec)[0])
        except Exception as e:
            logger.warning(f"LightGBM scoring failed, falling back: {e}")
            return weighted_score(features)

    def _apply_diversity(self, ranked: List[Dict], diversity_window: int = 3) -> List[Dict]:
        """
        Reorders songs to prevent genre monotony using a sliding window penalty.
        Songs are moved down if their genre appeared in the last diversity_window picks.
        """
        if len(ranked) <= diversity_window:
            return ranked

        result = []
        recent_genres = []
        remaining = list(ranked)

        while remaining and len(result) < len(ranked):
            placed = False
            for i, song in enumerate(remaining):
                genre = song.get("genre", "")
                genre_count = recent_genres[-diversity_window:].count(genre)
                if genre_count < 2:
                    result.append(song)
                    remaining.pop(i)
                    recent_genres.append(genre)
                    placed = True
                    break
            if not placed:
                # Force-place the top remaining song to avoid infinite loop
                result.append(remaining.pop(0))
                recent_genres.append(result[-1].get("genre", ""))

        return result

    def _inject_exploration(
        self,
        ranked: List[Dict],
        all_scored: List[Dict],
        genres_heard: set,
        all_genres: List[str],
        explore_slots: int,
    ) -> List[Dict]:
        """
        Forces `explore_slots` songs from unfamiliar genres into positions 4-7 of the slate.
        Ensures users always discover something new without overwhelming them.
        """
        if explore_slots <= 0 or not all_genres:
            return ranked

        unexplored = [g for g in all_genres if g not in genres_heard]
        if not unexplored:
            return ranked

        exploration_picks = []
        used_genres = set()

        for song in all_scored:
            if len(exploration_picks) >= explore_slots:
                break
            genre = song.get("genre", "")
            if genre in unexplored and genre not in used_genres:
                song_copy = {**song, "rec_reason": "Explore New", "rec_reason_icon": "fa-compass",
                             "rec_reason_color": "#45aaf2", "rec_category": "explore_new"}
                exploration_picks.append(song_copy)
                used_genres.add(genre)

        if not exploration_picks:
            return ranked

        # Remove exploration songs from ranked if they're already there
        exp_ids = {s["song_id"] for s in exploration_picks}
        ranked_filtered = [s for s in ranked if s["song_id"] not in exp_ids]

        # Inject at positions 4-6 (after top picks, before the rest)
        inject_at = min(4, len(ranked_filtered))
        final = (
            ranked_filtered[:inject_at]
            + exploration_picks
            + ranked_filtered[inject_at:]
        )
        return final

    def get_context_info(self, context_type: Optional[str] = None) -> Dict[str, Any]:
        """Returns context metadata for a given context type."""
        ct = context_type or infer_context_from_hour()
        return {"context_type": ct, **get_context_profile(ct)}


# ── Singleton ─────────────────────────────────────────────────────────────────

hybrid_ranker = HybridRanker()
