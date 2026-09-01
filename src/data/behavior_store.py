"""
BehaviorStore — Real-Time User Behavior Ledger.

Collects and aggregates:
- Likes, dislikes, skips (early / late), replays, playlist saves
- Listening time per song (milliseconds)
- Search history (queries + clicked song IDs)
- Recent activity with exponential recency-decay weighting

All data is stored in-memory (per server process) and reset on restart.
Designed as a singleton shared across all API requests.
"""

import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ─── Constants ────────────────────────────────────────────────────────────────

RECENCY_DECAY_LAMBDA = 0.85   # Multiplicative decay per event position (most-recent = 1.0)
MAX_HISTORY_PER_USER = 200    # Maximum events kept per user
MAX_SEARCH_HISTORY   = 20     # Search queries remembered per user
REPLAY_THRESHOLD_PCT = 0.80   # Fraction of song listened before seeking back = replay


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BehaviorEvent:
    """A single user interaction event."""
    event_type: str            # liked | disliked | no_skip | skip_early | skip_late
                               # | replay | playlist_save | search_click | listen_end
    song_id: str
    title: str
    artist_name: str
    genre: str
    listen_ms: int = 0         # milliseconds actually listened
    duration_ms: int = 210000  # total song duration
    energy: float = 0.65
    valence: float = 0.60
    danceability: float = 0.60
    tempo: float = 120.0
    timestamp: float = field(default_factory=time.time)
    search_query: Optional[str] = None  # if event_type == search_click


@dataclass
class UserBehaviorProfile:
    """Aggregated real-time profile derived from BehaviorEvents."""
    user_id: str
    # Per-genre accumulated weights (positive = liked, negative = skipped)
    genre_weights: Dict[str, float] = field(default_factory=dict)
    # Per-artist accumulated weights
    artist_weights: Dict[str, float] = field(default_factory=dict)
    # Listening time per song_id (ms)
    listen_time_ms: Dict[str, int] = field(default_factory=dict)
    # Replay counts per song_id
    replay_counts: Dict[str, int] = field(default_factory=dict)
    # Recent song IDs in reverse-chronological order
    recent_song_ids: List[str] = field(default_factory=list)
    # Liked song IDs
    liked_song_ids: List[str] = field(default_factory=list)
    # Skipped song IDs
    skipped_song_ids: List[str] = field(default_factory=list)
    # Mean energy over listened songs (running avg)
    avg_energy: float = 0.65
    avg_valence: float = 0.60
    avg_danceability: float = 0.60
    avg_tempo: float = 120.0
    total_events: int = 0
    # Search terms typed this session
    search_history: List[str] = field(default_factory=list)


# ─── BehaviorStore ────────────────────────────────────────────────────────────

class BehaviorStore:
    """
    Central in-memory store for all user behavior signals.
    Thread-safe for single-worker FastAPI (uses Python GIL; add asyncio.Lock for multi-worker).
    """

    def __init__(self):
        # Raw event log per user (bounded deque)
        self._events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_PER_USER))
        # Derived profiles (updated incrementally)
        self._profiles: Dict[str, UserBehaviorProfile] = {}
        # Search history per user
        self._search_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_SEARCH_HISTORY))

    # ── Public API ────────────────────────────────────────────────────────────

    def record_event(
        self,
        user_id: str,
        song_id: str,
        event_type: str,
        title: str = "",
        artist_name: str = "",
        genre: str = "Pop",
        listen_ms: int = 0,
        duration_ms: int = 210000,
        energy: float = 0.65,
        valence: float = 0.60,
        danceability: float = 0.60,
        tempo: float = 120.0,
        search_query: Optional[str] = None,
    ) -> None:
        """Records a single user interaction event and updates the derived profile."""
        evt = BehaviorEvent(
            event_type=event_type,
            song_id=song_id,
            title=title,
            artist_name=artist_name,
            genre=genre,
            listen_ms=listen_ms,
            duration_ms=duration_ms,
            energy=energy,
            valence=valence,
            danceability=danceability,
            tempo=tempo,
            search_query=search_query,
        )
        self._events[user_id].appendleft(evt)  # most-recent first
        self._update_profile(user_id, evt)

        # Track search history separately
        if event_type == "search_click" and search_query:
            self._search_history[user_id].appendleft(search_query)

    def get_profile(self, user_id: str) -> UserBehaviorProfile:
        """Returns the current aggregated behavior profile for a user."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserBehaviorProfile(user_id=user_id)
        return self._profiles[user_id]

    def get_recent_song_ids(self, user_id: str, n: int = 20) -> List[str]:
        """Returns the N most recently interacted song IDs."""
        events = list(self._events.get(user_id, []))
        seen = set()
        result = []
        for evt in events:
            if evt.song_id not in seen:
                seen.add(evt.song_id)
                result.append(evt.song_id)
                if len(result) >= n:
                    break
        return result

    def get_search_history(self, user_id: str, n: int = 5) -> List[str]:
        """Returns the N most recent search queries."""
        return list(self._search_history.get(user_id, []))[:n]

    def get_recency_weighted_genre_preferences(self, user_id: str) -> Dict[str, float]:
        """
        Returns genre preference weights with exponential recency decay.
        Most recent events contribute most; older events decay by RECENCY_DECAY_LAMBDA.
        """
        events = list(self._events.get(user_id, []))
        genre_scores: Dict[str, float] = defaultdict(float)

        for i, evt in enumerate(events):
            weight = (RECENCY_DECAY_LAMBDA ** i) * self._event_signal(evt.event_type)
            genre_scores[evt.genre] += weight

        # Normalize
        total = sum(abs(v) for v in genre_scores.values()) or 1.0
        return {g: round(v / total, 4) for g, v in genre_scores.items()}

    def get_recency_weighted_artist_preferences(self, user_id: str) -> Dict[str, float]:
        """Returns artist preference weights with exponential recency decay."""
        events = list(self._events.get(user_id, []))
        artist_scores: Dict[str, float] = defaultdict(float)

        for i, evt in enumerate(events):
            weight = (RECENCY_DECAY_LAMBDA ** i) * self._event_signal(evt.event_type)
            artist_scores[evt.artist_name] += weight

        total = sum(abs(v) for v in artist_scores.values()) or 1.0
        return {a: round(v / total, 4) for a, v in artist_scores.items()}

    def get_preferred_acoustic_centroid(self, user_id: str) -> Dict[str, float]:
        """
        Returns the recency-weighted acoustic centroid (mean features) from liked/full-listened songs.
        Used to compute acoustic deltas for ranking candidates.
        """
        events = list(self._events.get(user_id, []))
        if not events:
            return {"energy": 0.65, "valence": 0.60, "danceability": 0.60, "tempo": 120.0}

        w_sum = 0.0
        e_sum = v_sum = d_sum = t_sum = 0.0

        for i, evt in enumerate(events):
            sig = self._event_signal(evt.event_type)
            if sig <= 0:
                continue  # Only positive signals for centroid
            w = (RECENCY_DECAY_LAMBDA ** i) * sig
            e_sum += w * evt.energy
            v_sum += w * evt.valence
            d_sum += w * evt.danceability
            t_sum += w * evt.tempo
            w_sum += w

        if w_sum < 1e-6:
            return {"energy": 0.65, "valence": 0.60, "danceability": 0.60, "tempo": 120.0}

        return {
            "energy": round(e_sum / w_sum, 4),
            "valence": round(v_sum / w_sum, 4),
            "danceability": round(d_sum / w_sum, 4),
            "tempo": round(t_sum / w_sum, 2),
        }

    def get_top_liked_genres(self, user_id: str, n: int = 3) -> List[str]:
        """Returns top N positively weighted genres."""
        genre_prefs = self.get_recency_weighted_genre_preferences(user_id)
        positive = {g: w for g, w in genre_prefs.items() if w > 0}
        return sorted(positive, key=positive.get, reverse=True)[:n]

    def get_top_liked_artists(self, user_id: str, n: int = 3) -> List[str]:
        """Returns top N positively weighted artists."""
        artist_prefs = self.get_recency_weighted_artist_preferences(user_id)
        positive = {a: w for a, w in artist_prefs.items() if w > 0}
        return sorted(positive, key=positive.get, reverse=True)[:n]

    def get_skipped_genres(self, user_id: str) -> List[str]:
        """Returns genres where skip signal dominates."""
        genre_prefs = self.get_recency_weighted_genre_preferences(user_id)
        return [g for g, w in genre_prefs.items() if w < -0.05]

    def get_unexplored_genres(self, user_id: str, all_genres: List[str]) -> List[str]:
        """Returns genres the user has never interacted with — candidates for exploration."""
        events = list(self._events.get(user_id, []))
        heard_genres = {evt.genre for evt in events}
        return [g for g in all_genres if g not in heard_genres]

    def compute_session_mood(self, user_id: str, n: int = 10) -> str:
        """
        Infers current mood from recent valence average.
        Returns: 'happy' | 'melancholic' | 'neutral'
        """
        events = list(self._events.get(user_id, []))[:n]
        if not events:
            return "neutral"
        avg_v = sum(e.valence for e in events) / len(events)
        if avg_v > 0.65:
            return "happy"
        elif avg_v < 0.40:
            return "melancholic"
        return "neutral"

    def compute_session_energy_trend(self, user_id: str, n: int = 10) -> str:
        """
        Infers energy trend from recent energy values.
        Returns: 'high' | 'low' | 'mixed'
        """
        events = list(self._events.get(user_id, []))[:n]
        if not events:
            return "mixed"
        avg_e = sum(e.energy for e in events) / len(events)
        if avg_e > 0.68:
            return "high"
        elif avg_e < 0.42:
            return "low"
        return "mixed"

    def get_replay_song_ids(self, user_id: str) -> List[str]:
        """Returns song IDs that were replayed at least once."""
        profile = self.get_profile(user_id)
        return [sid for sid, cnt in profile.replay_counts.items() if cnt >= 1]

    def get_full_profile_dict(self, user_id: str) -> Dict[str, Any]:
        """Returns a serializable snapshot of the full behavior profile."""
        profile = self.get_profile(user_id)
        genre_prefs  = self.get_recency_weighted_genre_preferences(user_id)
        artist_prefs = self.get_recency_weighted_artist_preferences(user_id)
        centroid     = self.get_preferred_acoustic_centroid(user_id)
        mood         = self.compute_session_mood(user_id)
        energy_trend = self.compute_session_energy_trend(user_id)

        # Valence trajectory (last 15 events)
        events = list(self._events.get(user_id, []))[:15]
        valence_trajectory = [
            {"event_type": e.event_type, "valence": round(e.valence, 3), "song_id": e.song_id}
            for e in reversed(events)
        ]

        # Listen time summary
        top_listened = sorted(
            profile.listen_time_ms.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "user_id": user_id,
            "total_events": profile.total_events,
            "genre_weights": genre_prefs,
            "artist_weights": {a: w for a, w in list(artist_prefs.items())[:10]},
            "acoustic_centroid": centroid,
            "session_mood": mood,
            "energy_trend": energy_trend,
            "liked_count": len(profile.liked_song_ids),
            "skipped_count": len(profile.skipped_song_ids),
            "replay_songs": list(profile.replay_counts.keys()),
            "top_listened_songs": [{"song_id": sid, "listen_ms": ms} for sid, ms in top_listened],
            "recent_song_ids": profile.recent_song_ids[:20],
            "search_history": self.get_search_history(user_id),
            "valence_trajectory": valence_trajectory,
        }

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _event_signal(self, event_type: str) -> float:
        """Maps event type to a numeric reward signal for weighting."""
        return {
            "liked": 2.0,
            "playlist_save": 1.8,
            "replay": 1.5,
            "no_skip": 1.0,
            "listen_end": 0.8,
            "search_click": 0.5,
            "skip_late": -0.5,
            "skip_early": -1.0,
            "disliked": -2.0,
        }.get(event_type, 0.0)

    def _update_profile(self, user_id: str, evt: BehaviorEvent) -> None:
        """Incrementally updates the derived UserBehaviorProfile from a new event."""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserBehaviorProfile(user_id=user_id)

        p = self._profiles[user_id]
        p.total_events += 1

        sig = self._event_signal(evt.event_type)

        # Genre & artist weights
        p.genre_weights[evt.genre]       = p.genre_weights.get(evt.genre, 0.0) + sig
        p.artist_weights[evt.artist_name] = p.artist_weights.get(evt.artist_name, 0.0) + sig

        # Listen time accumulation
        if evt.listen_ms > 0:
            p.listen_time_ms[evt.song_id] = p.listen_time_ms.get(evt.song_id, 0) + evt.listen_ms

        # Replay counts
        if evt.event_type == "replay":
            p.replay_counts[evt.song_id] = p.replay_counts.get(evt.song_id, 0) + 1

        # Liked / Skipped tracking
        if evt.event_type in ("liked", "playlist_save"):
            if evt.song_id not in p.liked_song_ids:
                p.liked_song_ids.insert(0, evt.song_id)
        elif evt.event_type in ("skip_early", "disliked"):
            if evt.song_id not in p.skipped_song_ids:
                p.skipped_song_ids.append(evt.song_id)

        # Recent song IDs (deduplicated)
        if evt.song_id in p.recent_song_ids:
            p.recent_song_ids.remove(evt.song_id)
        p.recent_song_ids.insert(0, evt.song_id)
        p.recent_song_ids = p.recent_song_ids[:MAX_HISTORY_PER_USER]

        # Running acoustic averages (exponential moving average)
        alpha = 0.15
        p.avg_energy      = (1 - alpha) * p.avg_energy      + alpha * evt.energy
        p.avg_valence     = (1 - alpha) * p.avg_valence     + alpha * evt.valence
        p.avg_danceability= (1 - alpha) * p.avg_danceability+ alpha * evt.danceability
        p.avg_tempo       = (1 - alpha) * p.avg_tempo       + alpha * evt.tempo

        # Search history
        if evt.event_type == "search_click" and evt.search_query:
            if evt.search_query not in p.search_history:
                p.search_history.insert(0, evt.search_query)
            p.search_history = p.search_history[:MAX_SEARCH_HISTORY]


# ── Singleton ────────────────────────────────────────────────────────────────

behavior_store = BehaviorStore()
