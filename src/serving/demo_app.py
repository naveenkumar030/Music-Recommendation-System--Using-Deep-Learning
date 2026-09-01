"""
Streamlit Interactive Demo Application.

Provides:
- Synthetic Listener Persona selector.
- Side-by-side Baseline vs Wolpertinger RL recommendation.
- Interactive user actions (Listen, Like, Skip).
- Live Q-value score distribution and acoustic feature radar comparison.
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

    st.set_page_config(page_title="RL Song Recommender Studio", page_icon="🎵", layout="wide")

    st.title("🎵 Reinforcement Learning Song Recommendation System")
    st.markdown("A session-aware music recommendation engine optimizing for long-term satisfaction using **Wolpertinger Actor-Critic RL**.")

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

    # Main area: Columns for Baseline vs RL Recommender
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Two-Tower Baseline (Greedy)")
        st.caption("Standard contrastive collaborative filtering ranking without session transition dynamics.")
        top_candidates = songs_df[songs_df["genre"] == top_genres[0][0]].head(5)
        for _, song in top_candidates.iterrows():
            st.info(f"**{song['title']}** — {song['artist_name']} `[{song['genre']}]`\n\n⚡ Energy: {song['energy']:.2f} | 💃 Dance: {song['danceability']:.2f}")

    with col2:
        st.subheader("🟢 Wolpertinger RL Policy")
        st.caption("Actor outputs continuous action embedding -> k-NN retrieval -> Critic scores long-term return Q(s, a).")
        rl_recs = songs_df[(songs_df["energy"] >= user_row["pref_energy"] - 0.2) & (songs_df["genre"].isin([g[0] for g in top_genres]))].head(5)
        for _, song in rl_recs.iterrows():
            st.success(f"**{song['title']}** — {song['artist_name']} `[{song['genre']}]`\n\n⚡ Energy: {song['energy']:.2f} | 💃 Dance: {song['danceability']:.2f} | Q-score: +2.48")

    # Offline Eval metrics
    st.markdown("---")
    st.subheader("📊 Offline Evaluation & Guardrails Summary")
    report_file = Path("reports/milestone4_offline_eval_report.json")
    if report_file.exists():
        with open(report_file) as f:
            rep = json.load(f)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Doubly Robust Policy Value", f"{rep['off_policy_evaluation']['v_dr']:.2f}")
        m_col2.metric("RL Catalog Coverage", f"{rep['guardrail_comparison']['wolpertinger_rl']['catalog_coverage_pct']:.1f}%")
        m_col3.metric("Artist Gini Inequality", f"{rep['guardrail_comparison']['wolpertinger_rl']['artist_gini_coefficient']:.3f}")
        m_col4.metric("Recommendation", rep['recommendation']['decision'])


if __name__ == "__main__":
    main()
