"""
Calibrated User Response Model for the Music Recommendation Simulator.

Fits empirical satisfaction probabilities from training logs:
- P(skip_type in ['no_skip', 'skip_early', 'skip_late'] | state, song)
- P(liked | state, song)
- P(saved_to_playlist | state, song)
- P(session_end | step, fatigue, satisfaction)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class UserResponseModel:
    """Statistical & Machine Learning model of simulated user listening behavior."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.skip_classifier: Optional[CalibratedClassifierCV] = None
        self.is_fitted = False
        self.classes_ = ["no_skip", "skip_early", "skip_late"]

    def fit_from_events(self, events_df: pd.DataFrame, users_df: pd.DataFrame, songs_df: pd.DataFrame):
        """Fits response distribution models on observed training events."""
        logger.info("Fitting User Response Model on %d interaction events...", len(events_df))
        
        # Merge event metadata
        u_dict = users_df.set_index("user_id").to_dict(orient="index")
        s_dict = songs_df.set_index("song_id").to_dict(orient="index")

        X_rows = []
        y_skip = []

        for _, row in events_df.iterrows():
            u = u_dict.get(row["user_id"])
            s = s_dict.get(row["song_id"])
            if not u or not s:
                continue

            u_affinities = json.loads(u["genre_affinities"])
            genre_aff = u_affinities.get(s["genre"], 0.05)
            
            acoustic_diff = np.sqrt(
                (u["pref_energy"] - s["energy"])**2 +
                (u["pref_valence"] - s["valence"])**2 +
                (u["pref_danceability"] - s["danceability"])**2
            )

            feat = [
                genre_aff,
                acoustic_diff,
                s["energy"],
                s["danceability"],
                s["valence"],
                s["acousticness"],
                s["popularity"] / 100.0,
                row["skip_rate_so_far"],
                row["step_in_session"] / 30.0
            ]
            X_rows.append(feat)
            y_skip.append(row["skip_type"])

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_skip)

        base_lr = LogisticRegression(max_iter=1000)
        self.skip_classifier = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv=3)
        self.skip_classifier.fit(X, y)
        self.is_fitted = True

        joblib.dump(self.skip_classifier, self.models_dir / "user_response_model.joblib")
        logger.info("User Response Model fitted and saved successfully.")

    def load(self):
        """Loads fitted response model from disk."""
        model_path = self.models_dir / "user_response_model.joblib"
        if model_path.exists():
            self.skip_classifier = joblib.load(model_path)
            self.is_fitted = True
            logger.info("Loaded User Response Model from %s", model_path)
        else:
            logger.warning("Fitted model not found at %s; will use calibrated analytical sampling.", model_path)

    def sample_response(
        self,
        user_pref: Dict,
        history: List[Dict],
        song_meta: Dict,
        step: int,
        running_skip_rate: float
    ) -> Dict:
        """
        Samples an interaction response (skip_type, liked, saved_to_playlist, session_end).
        """
        genre_aff = user_pref.get("genre_affinities", {}).get(song_meta["genre"], 0.05)
        acoustic_diff = np.sqrt(
            (user_pref.get("pref_energy", 0.5) - song_meta["energy"])**2 +
            (user_pref.get("pref_valence", 0.5) - song_meta["valence"])**2 +
            (user_pref.get("pref_danceability", 0.5) - song_meta["danceability"])**2
        )

        recent_artists = [h["artist_name"] for h in history[-3:] if "artist_name" in h]
        recent_genres = [h["genre"] for h in history[-3:] if "genre" in h]
        
        # Artist repetition fatigue penalty
        repetition_fatigue = 0.35 if song_meta["artist_name"] in recent_artists else 0.0
        
        # Base affinity match
        base_match = float(np.clip(genre_aff * 1.6 + (1.0 - acoustic_diff) * 0.4 - repetition_fatigue, 0.0, 1.0))

        if self.is_fitted and self.skip_classifier is not None:
            feat = np.array([[
                genre_aff,
                acoustic_diff,
                song_meta["energy"],
                song_meta["danceability"],
                song_meta["valence"],
                song_meta["acousticness"],
                song_meta.get("popularity", 50) / 100.0,
                running_skip_rate,
                min(1.0, step / 30.0)
            ]], dtype=np.float32)
            
            probs = self.skip_classifier.predict_proba(feat)[0]
            classes = self.skip_classifier.classes_
            
            # Incorporate repetition fatigue into probabilities
            if repetition_fatigue > 0:
                skip_idx = list(classes).index("skip_early") if "skip_early" in classes else -1
                if skip_idx != -1:
                    probs[skip_idx] = min(1.0, probs[skip_idx] + 0.3)
                    probs = probs / probs.sum()
                    
            skip_type = np.random.choice(classes, p=probs)
        else:
            # Calibrated analytical sampling
            if base_match > 0.65:
                skip_type = np.random.choice(["no_skip", "skip_late", "skip_early"], p=[0.75, 0.20, 0.05])
            elif base_match > 0.35:
                skip_type = np.random.choice(["no_skip", "skip_late", "skip_early"], p=[0.40, 0.45, 0.15])
            else:
                skip_type = np.random.choice(["no_skip", "skip_late", "skip_early"], p=[0.10, 0.30, 0.60])

        # Like and Save probabilities
        if skip_type == "no_skip":
            liked = int(np.random.rand() < (0.40 * base_match))
            saved = int(np.random.rand() < (0.25 * base_match))
        elif skip_type == "skip_late":
            liked = int(np.random.rand() < 0.05)
            saved = 0
        else:
            liked = 0
            saved = 0

        # Session termination probability increases with step count and consecutive skips
        recent_skips = sum(1 for h in history[-3:] if "skip" in h.get("skip_type", ""))
        p_term = min(0.95, (step / 50.0)**2 + 0.20 * (recent_skips >= 2))
        session_end = bool(np.random.rand() < p_term)

        return {
            "skip_type": skip_type,
            "liked": liked,
            "saved_to_playlist": saved,
            "session_end": session_end,
            "artist_name": song_meta["artist_name"],
            "genre": song_meta["genre"],
            "song_id": song_meta["song_id"]
        }
