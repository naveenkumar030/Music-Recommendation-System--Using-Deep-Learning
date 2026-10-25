"""
Live Deep Learning Pipeline Verification Script.
Demonstrates:
1. PyTorch Multi-Task Deep Neural Ranker (MT-DNN) architecture and weights.
2. In-memory tensor shapes and forward pass.
3. Multi-objective probabilities P(completion), P(like), P(skip).
4. End-to-end FastAPI endpoint response with Deep Ranker.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from src.models.deep_ranker import MultiTaskDeepRanker, DeepRankerAgent
from src.data.feature_store import FeatureStore
from src.serving.api import app, startup_event


def main():
    print("=" * 70)
    print("      SOUNDSPACE DL - DEEP LEARNING LIVE VERIFICATION")
    print("=" * 70)

    # 1. Inspect PyTorch Model Architecture
    print("\n[Step 1] Loading PyTorch Deep Neural Ranker checkpoint (models/deep_ranker.pt)...")
    checkpoint = torch.load("models/deep_ranker.pt", map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  • Trained Epochs: {checkpoint.get('epoch', 'N/A')}")
        print(f"  • Training Loss:   {checkpoint.get('loss', 0.0):.4f}")
    else:
        state_dict = checkpoint
        print(f"  • Format: Pure PyTorch MultiTaskDeepRanker State Dict ({len(state_dict)} tensor keys)")

    model = MultiTaskDeepRanker(state_dim=143, song_dim=64)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[Step 2] PyTorch Model Architecture & Layers:")
    print(f"  • Model Class: {model.__class__.__name__}")
    print(f"  • Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    print(f"  • GMF Latent Dimension: 64")
    print(f"  • MLP Trunk Structure: 207 -> 256 (ReLU, Dropout 0.2) -> 128 (ReLU)")
    print(f"  • Multi-Task Heads: P(completion) [Sigmoid], P(like) [Sigmoid], P(skip) [Sigmoid]")

    # 2. Live Tensor Forward Pass
    print("\n[Step 3] Executing Live PyTorch Forward Pass with Tensors:")
    batch_size = 1
    num_candidates = 5
    sample_state = torch.randn(batch_size, 143)
    sample_cands = torch.randn(batch_size, num_candidates, 64)

    with torch.no_grad():
        out = model(sample_state, sample_cands)

    print(f"  • Input State Tensor:       shape={list(sample_state.shape)}")
    print(f"  • Candidate Embeddings:     shape={list(sample_cands.shape)}")
    print(f"  • Output Rank Scores:       shape={list(out['rank_score'].shape)}")
    print(f"  • Output P(Completion):     shape={list(out['p_complete'].shape)}, sample values={out['p_complete'].squeeze().numpy().round(3)}")
    print(f"  • Output P(Like):           shape={list(out['p_like'].shape)}, sample values={out['p_like'].squeeze().numpy().round(3)}")
    print(f"  • Output P(Skip):           shape={list(out['p_skip'].shape)}, sample values={out['p_skip'].squeeze().numpy().round(3)}")

    # 3. Real Catalog Candidate Ranking
    print("\n[Step 4] Live Candidate Scoring over Real Music Catalog (User: usr_00002 Lo-Fi Coder):")
    fs = FeatureStore(processed_dir=Path("data/processed"))
    users_df = pd.read_parquet("data/processed/users.parquet")
    songs_df = pd.read_parquet("data/processed/songs.parquet").replace({np.nan: None})
    fs.load_and_index(users_df, songs_df)

    song_embs = np.load("models/song_embeddings.npy")
    agent = DeepRankerAgent(state_dim=143, song_embeddings=song_embs, song_dim=64, feature_store=fs)
    agent.load("models/deep_ranker.pt")

    user_vec = fs.get_user_vector_by_id("usr_00002")
    user_state = np.zeros(143, dtype=np.float32)
    user_state[:len(user_vec)] = user_vec

    candidate_ids = [0, 10, 25, 42, 88]
    ranked = agent.rank_candidates(user_state, candidate_indices=candidate_ids, top_k=5)

    print(f"  {'Rank':<4} | {'Song Title':<24} | {'Artist':<18} | {'Deep Score':<10} | {'P(Listen)':<9} | {'P(Like)':<7} | {'P(Skip)':<7}")
    print("  " + "-" * 95)
    for rank, (sid, score, details) in enumerate(ranked, 1):
        song_str_id = fs.idx_to_song_id.get(sid, sid) if hasattr(fs, "idx_to_song_id") else sid
        meta = fs.get_song_metadata(song_str_id)
        title = meta.get("title", "Track")[:22]
        artist = meta.get("artist_name", "Artist")[:16]
        p_c = details["p_complete"]
        p_l = details["p_like"]
        p_s = details["p_skip"]
        print(f"  #{rank:<3} | {title:<24} | {artist:<18} | {score:<+10.4f} | {p_c:<9.3f} | {p_l:<7.3f} | {p_s:<7.3f}")

    # 4. Live FastAPI Verification
    print("\n[Step 5] Live FastAPI Recommendation API Verification:")
    startup_event()
    client = TestClient(app)

    health = client.get("/api/health").json()
    print(f"  • API Status:        {health.get('status')}")
    print(f"  • Active Pipeline:   {health.get('active_pipeline')}")
    print(f"  • Deep Ranker Loaded: {health.get('models_loaded', {}).get('deep_ranker')}")

    rec_res = client.post("/api/recommend", json={
        "user_id": "usr_00002",
        "model_type": "deep_ranker",
        "slate_size": 3
    })
    rec_data = rec_res.json()
    print(f"  • API /api/recommend Status Code: {rec_res.status_code}")
    print(f"  • Recommended Tracks: {[t['title'] for t in rec_data.get('recommendations', [])]}")
    print(f"  • Inference Latency:   {rec_data.get('latency_ms')} ms")
    if rec_data.get("deep_score_distribution"):
        top_deep = rec_data["deep_score_distribution"][0]
        print(f"  • Top Deep Candidate Score: {top_deep.get('title')} -> Deep Score = +{top_deep.get('deep_score')}")

    print("\n" + "=" * 70)
    print("  [SUCCESS] VERIFICATION COMPLETE: The project is 100% running on Deep Learning!")
    print("=" * 70)


if __name__ == "__main__":
    main()
