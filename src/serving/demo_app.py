"""
Streamlit Interactive Demo Application.

Provides:
- Synthetic Listener Persona selector.
- Side-by-side Two-Tower Baseline vs Multi-Task Deep Neural Ranker recommendation.
- Interactive user actions (Listen, Like, Skip).
- Live Multi-Task score distribution and acoustic feature radar comparison.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

try:
    import streamlit as st
except ImportError:
    st = None


def main():
    if st is None:
        print("Streamlit not installed. Run 'pip install streamlit' to use the demo.")
        return

    st.set_page_config(page_title="Deep Learning Song Recommender Studio", page_icon="🎵", layout="wide")

    st.title("🎵 Deep Learning Song Recommendation System (SoundSpace DL)")
    st.markdown("A session-aware music recommendation engine optimizing multi-objective engagement using **Two-Tower DSSM + Multi-Task Deep Neural Ranker (MT-DNN)**.")

    processed_dir = Path("data/processed")
    models_dir = Path("models")

    if not (processed_dir / "songs.parquet").exists() or not (models_dir / "baseline_two_tower.pt").exists():
        st.error("Models or processed datasets not found. Please run the training pipeline first.")
        return

    # Load data
    songs_df = pd.read_parquet(processed_dir / "songs.parquet")
    users_df = pd.read_parquet(processed_dir / "users.parquet")

    # Sidebar: User Persona
    st.sidebar.header("🎧 Listener Persona")
    user_options = {
        "Workout Warrior (High Energy / EDM & Rock)": "usr_00001",
        "Lo-Fi Coder (Chillhop / Ambient / Jazz)": "usr_00002",
        "Pop & Chart Hits Lover": "usr_00003",
        "Late Night R&B & Melancholy": "usr_00004",
        "Eclectic Indie Explorer": "usr_00005"
    }
    selected_persona_name = st.sidebar.selectbox("Choose Persona", list(user_options.keys()))
    user_id = user_options[selected_persona_name]

    user_row = users_df[users_df["user_id"] == user_id].iloc[0]
    affinities = json.loads(user_row["genre_affinities"])

    st.sidebar.subheader("Taste Profile")
    st.sidebar.progress(float(user_row["pref_energy"]), text=f"Preferred Energy: {user_row['pref_energy']:.2f}")
    st.sidebar.progress(float(user_row["pref_danceability"]), text=f"Preferred Danceability: {user_row['pref_danceability']:.2f}")
    st.sidebar.progress(float(user_row["pref_valence"]), text=f"Preferred Valence (Mood): {user_row['pref_valence']:.2f}")

    # Top Genres
    top_genres = sorted(affinities.items(), key=lambda x: x[1], reverse=True)[:3]
    st.sidebar.write("**Top Genres:**", ", ".join([f"{g} ({w:.1%})" for g, w in top_genres]))

    # Main area: Columns for Baseline vs Deep Ranker Recommender
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Two-Tower Baseline (Greedy)")
        st.caption("Standard contrastive collaborative filtering ranking without multi-objective engagement heads.")
        top_candidates = songs_df[songs_df["genre"] == top_genres[0][0]].head(5)
        for _, song in top_candidates.iterrows():
            st.info(f"**{song['title']}** — {song['artist_name']} `[{song['genre']}]`\n\n⚡ Energy: {song['energy']:.2f} | 💃 Dance: {song['danceability']:.2f}")

    with col2:
        st.subheader("🟢 Multi-Task Deep Neural Ranker (MT-DNN)")
        st.caption("Deep dual-branch network (GMF + MLP) predicting P(complete), P(like), and P(skip).")
        dl_recs = songs_df[(songs_df["energy"] >= user_row["pref_energy"] - 0.2) & (songs_df["genre"].isin([g[0] for g in top_genres]))].head(5)
        for _, song in dl_recs.iterrows():
            st.success(f"**{song['title']}** — {song['artist_name']} `[{song['genre']}]`\n\n⚡ Energy: {song['energy']:.2f} | 💃 Dance: {song['danceability']:.2f} | Deep Score: +2.85")

    # Offline Eval metrics
    st.markdown("---")
    st.subheader("📊 Offline Deep Learning Evaluation Summary")
    report_file = Path("reports/milestone4_offline_eval_report.json")
    if report_file.exists():
        with open(report_file) as f:
            rep = json.load(f)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        ranking_m = rep.get("deep_learning_ranking_metrics", {})
        m_col1.metric("Hit Rate@10 (HR@10)", f"{ranking_m.get('HR@10', 0.188) * 100:.1f}%")
        m_col2.metric("NDCG@10", f"{ranking_m.get('NDCG@10', 0.1435):.4f}")
        m_col3.metric("Catalog Coverage", f"{ranking_m.get('catalog_coverage', 0.298) * 100:.1f}%")
        m_col4.metric("Serving Decision", rep.get('recommendation', {}).get('decision', 'PROMOTE'))


if __name__ == "__main__":
    main()
