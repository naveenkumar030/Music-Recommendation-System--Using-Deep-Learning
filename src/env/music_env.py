"""
Gymnasium Music Recommendation MDP Environment (SongRecEnv).

Implements the MDP formulation specified in Section 3 & Section 5 of PROJECT_PLAN.md:
- State Space: user_embedding, session_encoding, context vector, session_stats.
- Action Space: candidate track selection.
- Reward: skip penalties, like/save bonuses, fatigue penalty (-0.3), novelty bonus (+0.1).
- Transition: user satisfaction updates, session history accumulation.
"""

from dataclasses import dataclass
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.data.feature_store import FeatureStore
from .response_model import UserResponseModel


@dataclass
class PlaybackEvent:
    song_id: str
    artist: str
    genre: str
    skip_type: str        # 'no_skip', 'skip_early', 'skip_late'
    liked: int            # 0 or 1
    saved_to_playlist: int # 0 or 1
    session_end: bool = False


def genre_novelty_bonus(genre: str, recent_genres: List[str]) -> float:
    """Returns 0.0 if genre in recent_genres else 1.0."""
    return 0.0 if genre in recent_genres else 1.0


def compute_reward(event: PlaybackEvent, session_history: List[PlaybackEvent]) -> float:
    """
    Computes MDP reward following the RL recommendation formulation:
    - no_skip (Listen): +0.5
    - skip_early (Skip): -1.0
    - skip_late: -0.2
    - liked (Like): +1.0
    - saved_to_playlist: +0.5
    - repetition fatigue penalty: -0.3 (if same artist in last 3)
    - genre novelty bonus: +0.1 (if new genre)
    """
    r = 0.0
    if event.skip_type == "no_skip":
        r += 0.5
    elif event.skip_type == "skip_early":
        r -= 1.0
    elif event.skip_type == "skip_late":
        r -= 0.2

    r += 1.0 * event.liked
    r += 0.5 * event.saved_to_playlist

    recent = session_history[-3:]
    if any(e.artist == event.artist for e in recent):
        r -= 0.3  # repetition / fatigue penalty

    r += 0.1 * genre_novelty_bonus(event.genre, [e.genre for e in recent])
    return float(r)


class SongRecEnv(gym.Env):
    """
    Session-level music recommendation environment.
    Conforms to standard Gymnasium Environment specification.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_store: FeatureStore,
        response_model: Optional[UserResponseModel] = None,
        max_session_len: int = 50,
        history_window: int = 5
    ):
        super().__init__()
        self.feature_store = feature_store
        self.response_model = response_model or UserResponseModel()
        self.max_session_len = max_session_len
        self.history_window = history_window

        self.num_songs = self.feature_store.num_songs
        self.song_dim = self.feature_store.song_dim
        self.user_dim = self.feature_store.user_dim

        # Action space: integer song index (0 to num_songs - 1)
        self.action_space = spaces.Discrete(self.num_songs)

        # Flat state representation dimension:
        # user_dim + (history_window * (song_dim + 3)) + 3 (context) + 3 (session_stats)
        self.session_encoding_dim = self.history_window * (self.song_dim + 3)
        self.state_dim = self.user_dim + self.session_encoding_dim + 3 + 3

        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Internal episode variables
        self.current_user: Optional[Dict] = None
        self.current_user_id: Optional[str] = None
        self.history: List[PlaybackEvent] = []
        self.t = 0
        self.context: Dict[str, Any] = {}
        self.likes_so_far = 0
        self.skips_so_far = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Sample user from user pool or custom option
        if options and "user_id" in options:
            self.current_user_id = options["user_id"]
        else:
            random_idx = np.random.randint(0, self.feature_store.num_users)
            self.current_user_id = self.feature_store.idx_to_user_id[random_idx]

        user_row = self.feature_store.users_df[self.feature_store.users_df["user_id"] == self.current_user_id].iloc[0]
        self.current_user = {
            "user_id": self.current_user_id,
            "pref_energy": user_row["pref_energy"],
            "pref_valence": user_row["pref_valence"],
            "pref_danceability": user_row["pref_danceability"],
            "genre_affinities": json.loads(user_row["genre_affinities"])
        }

        self.history = []
        self.t = 0
        self.likes_so_far = 0
        self.skips_so_far = 0

        self.context = {
            "hour_of_day": int(np.random.randint(0, 24)),
            "day_of_week": int(np.random.randint(0, 7)),
            "device": np.random.choice(["mobile_ios", "mobile_android", "desktop_web"])
        }

        obs = self._get_state()
        info = {
            "user_id": self.current_user_id,
            "context": self.context,
            "step": self.t
        }
        return obs, info

    def step(self, action: Union[int, str]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, str):
            song_id = action
        else:
            action_idx = int(action)
            if action_idx in self.feature_store.idx_to_song_id:
                song_id = self.feature_store.idx_to_song_id[action_idx]
            else:
                song_id = self.feature_store.idx_to_song_id.get(0, self.feature_store.songs_df["song_id"].iloc[0])

        song_meta = self.feature_store.get_song_metadata(song_id)
        running_skip_rate = self.skips_so_far / max(1, self.t)

        # Sample user response from calibrated model
        history_dicts = [{"artist_name": e.artist, "genre": e.genre, "skip_type": e.skip_type} for e in self.history]
        resp = self.response_model.sample_response(
            user_pref=self.current_user,
            history=history_dicts,
            song_meta=song_meta,
            step=self.t,
            running_skip_rate=running_skip_rate
        )

        event = PlaybackEvent(
            song_id=song_id,
            artist=song_meta["artist_name"],
            genre=song_meta["genre"],
            skip_type=resp["skip_type"],
            liked=resp["liked"],
            saved_to_playlist=resp["saved_to_playlist"],
            session_end=resp["session_end"]
        )

        reward = compute_reward(event, self.history)
        self.history.append(event)
        self.t += 1

        if event.liked:
            self.likes_so_far += 1
        if "skip" in event.skip_type:
            self.skips_so_far += 1

        terminated = bool(event.session_end or self.t >= self.max_session_len)
        truncated = False

        obs = self._get_state()
        info = {
            "event": event,
            "step": self.t,
            "likes_so_far": self.likes_so_far,
            "skips_so_far": self.skips_so_far,
            "skip_rate": round(self.skips_so_far / self.t, 3),
            "song_id": song_id,
            "title": song_meta.get("title", ""),
            "artist": song_meta.get("artist_name", ""),
            "genre": song_meta.get("genre", "")
        }

        return obs, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Constructs flat observation vector matching MDP state specification."""
        u_vec = self.feature_store.get_user_vector_by_id(self.current_user_id)

        # Build sliding window session history encoding
        history_feats = []
        recent_events = self.history[-self.history_window:]
        
        # Pad if history is shorter than history_window
        padding_count = self.history_window - len(recent_events)
        for _ in range(padding_count):
            history_feats.extend([0.0] * (self.song_dim + 3))

        for ev in recent_events:
            s_vec = self.feature_store.get_song_vector_by_id(ev.song_id)
            # Response feedback representation: [is_no_skip, is_skip_early, liked]
            resp_vec = [
                1.0 if ev.skip_type == "no_skip" else 0.0,
                1.0 if ev.skip_type == "skip_early" else 0.0,
                1.0 if ev.liked else 0.0
            ]
            history_feats.extend(list(s_vec) + resp_vec)

        # Context features: hour normalized, day normalized, device flag
        ctx_vec = [
            self.context["hour_of_day"] / 24.0,
            self.context["day_of_week"] / 7.0,
            1.0 if "mobile" in self.context["device"] else 0.0
        ]

        # Session running stats
        skip_rate = self.skips_so_far / max(1, self.t)
        stats_vec = [
            skip_rate,
            self.likes_so_far / 10.0,
            self.t / float(self.max_session_len)
        ]

        full_state = np.concatenate([u_vec, np.array(history_feats, dtype=np.float32), np.array(ctx_vec, dtype=np.float32), np.array(stats_vec, dtype=np.float32)])
        return full_state.astype(np.float32)

    def get_user_profile(self) -> Dict:
        return dict(self.current_user)
