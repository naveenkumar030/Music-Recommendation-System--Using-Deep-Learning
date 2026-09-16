"""
Simulator Calibration and Validation Script.

Fits the UserResponseModel from data/processed/train_events.parquet and compares
aggregate simulated statistics (skip rate, like rate, session length) against empirical data.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.feature_store import FeatureStore
from src.env.response_model import UserResponseModel
from src.env.music_env import SongRecEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calibrate_and_validate_simulator(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports",
    num_sim_sessions: int = 300
) -> dict:
    p_dir = Path(processed_dir)
    r_dir = Path(reports_dir)
    r_dir.mkdir(parents=True, exist_ok=True)

    users_df = pd.read_parquet(p_dir / "users.parquet")
    songs_df = pd.read_parquet(p_dir / "songs.parquet")
    train_events = pd.read_parquet(p_dir / "train_events.parquet")

    # 1. Fit response model
    resp_model = UserResponseModel(models_dir=models_dir)
    resp_model.fit_from_events(train_events, users_df, songs_df)

    # 2. Compute Real Data Statistics
    real_skip_rates = train_events["skip_type"].value_counts(normalize=True).to_dict()
    real_like_rate = float(train_events["liked"].mean())
    real_session_lengths = train_events.groupby("session_id")["step_in_session"].count().tolist()

    real_stats = {
        "early_skip_rate": round(float(real_skip_rates.get("skip_early", 0.0)), 4),
        "late_skip_rate": round(float(real_skip_rates.get("skip_late", 0.0)), 4),
        "no_skip_rate": round(float(real_skip_rates.get("no_skip", 0.0)), 4),
        "like_rate": round(real_like_rate, 4),
        "mean_session_length": round(float(np.mean(real_session_lengths)), 2),
        "median_session_length": round(float(np.median(real_session_lengths)), 2)
    }

    # 3. Run Simulated Sessions
    feature_store = FeatureStore(processed_dir=processed_dir)
    feature_store.load_and_index(users_df, songs_df)

    env = SongRecEnv(feature_store=feature_store, response_model=resp_model, max_session_len=50)

    sim_events = []
    sim_session_lengths = []

    for s_idx in range(num_sim_sessions):
        obs, info = env.reset(seed=1000 + s_idx)
        done = False
        step_count = 0

        while not done:
            # Policy for simulation validation: sample action matching user preference cluster
            u_pref = env.get_user_profile()
            # Randomly select a song
            action_idx = np.random.randint(0, env.num_songs)
            obs, reward, terminated, truncated, step_info = env.step(action_idx)
            event = step_info["event"]
            sim_events.append(event)
            step_count += 1
            if terminated or truncated:
                done = True

        sim_session_lengths.append(step_count)

    sim_early_skips = sum(1 for e in sim_events if e.skip_type == "skip_early") / len(sim_events)
    sim_late_skips = sum(1 for e in sim_events if e.skip_type == "skip_late") / len(sim_events)
    sim_no_skips = sum(1 for e in sim_events if e.skip_type == "no_skip") / len(sim_events)
    sim_like_rate = sum(1 for e in sim_events if e.liked) / len(sim_events)

    sim_stats = {
        "early_skip_rate": round(float(sim_early_skips), 4),
        "late_skip_rate": round(float(sim_late_skips), 4),
        "no_skip_rate": round(float(sim_no_skips), 4),
        "like_rate": round(float(sim_like_rate), 4),
        "mean_session_length": round(float(np.mean(sim_session_lengths)), 2),
        "median_session_length": round(float(np.median(sim_session_lengths)), 2)
    }

    comparison_report = {
        "num_sim_sessions": num_sim_sessions,
        "total_sim_events": len(sim_events),
        "real_data_statistics": real_stats,
        "simulated_statistics": sim_stats
    }

    with open(r_dir / "simulator_calibration_report.json", "w") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info("Simulator Calibration & Validation Complete. Saved to %s", r_dir / "simulator_calibration_report.json")
    return comparison_report


if __name__ == "__main__":
    rep = calibrate_and_validate_simulator()
    print("\n--- Milestone 2 Simulator Calibration Report ---")
    print(json.dumps(rep, indent=2))
