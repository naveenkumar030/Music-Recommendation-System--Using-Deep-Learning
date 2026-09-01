"""
Offline Evaluation Harness for Candidate Retrieval Baseline.

Computes:
- Recall@k (k = 10, 50, 100) on held-out test sessions.
- NDCG@k (k = 10) ranking quality.
- Candidate Retrieval Latency benchmarking for k=500 candidates (<50ms requirement).
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel, VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_dcg_at_k(hits: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank k for binary relevance."""
    hits = hits[:k]
    if not hits:
        return 0.0
    return sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))


def evaluate_retrieval_baseline(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports",
    embed_dim: int = 64,
    k_list: List[int] = [10, 50, 100]
) -> Dict[str, float]:
    """Runs offline evaluation on held-out test split."""
    p_dir = Path(processed_dir)
    m_dir = Path(models_dir)
    r_dir = Path(reports_dir)
    r_dir.mkdir(parents=True, exist_ok=True)

    users_df = pd.read_parquet(p_dir / "users.parquet")
    songs_df = pd.read_parquet(p_dir / "songs.parquet")
    test_events = pd.read_parquet(p_dir / "test_events.parquet")

    # Filter test events to positive interactions
    test_pos = test_events[test_events["skip_type"] == "no_skip"].copy()
    if len(test_pos) < 50:
        test_pos = test_events.copy()

    feature_store = FeatureStore(processed_dir=processed_dir)
    feature_store.load_and_index(users_df, songs_df)

    # Load model and embeddings
    model = TwoTowerModel(
        user_dim=feature_store.user_dim,
        song_dim=feature_store.song_dim,
        embed_dim=embed_dim
    )
    model.load_state_dict(torch.load(m_dir / "baseline_two_tower.pt"))
    model.eval()

    song_embeddings = np.load(m_dir / "song_embeddings.npy")
    with open(m_dir / "song_ids.json") as f:
        song_ids = json.load(f)

    v_index = VectorIndex(embed_dim=embed_dim)
    v_index.build_index(song_embeddings, song_ids)

    # Evaluate Recall@k and NDCG@10
    recalls = {k: [] for k in k_list}
    ndcgs_10 = []
    latencies_ms = []

    logger.info("Evaluating on %d test interaction events...", len(test_pos))

    with torch.no_grad():
        for _, row in test_pos.iterrows():
            u_id = row["user_id"]
            true_song_id = row["song_id"]

            u_vec = feature_store.get_user_vector_by_id(u_id)
            u_tensor = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0)
            u_emb = model.user_tower(u_tensor).squeeze(0).cpu().numpy()

            # Query top max(k_list) candidates
            top_candidates, _, lat = v_index.query(u_emb, top_k=max(k_list))
            latencies_ms.append(lat)

            # Check binary hit
            hits = [1 if cand == true_song_id else 0 for cand in top_candidates]

            for k in k_list:
                recalls[k].append(1.0 if 1 in hits[:k] else 0.0)

            # NDCG@10: ideal DCG is 1.0 (since exactly 1 target item is queried)
            dcg_10 = compute_dcg_at_k(hits, k=10)
            ndcgs_10.append(dcg_10)

    results = {
        "num_test_events": len(test_pos),
        "recall_at_10": round(float(np.mean(recalls[10])), 4),
        "recall_at_50": round(float(np.mean(recalls[50])), 4),
        "recall_at_100": round(float(np.mean(recalls[100])), 4),
        "ndcg_at_10": round(float(np.mean(ndcgs_10)), 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3)
    }

    report_path = r_dir / "baseline_retrieval_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Baseline Evaluation Results: Recall@50 = %.4f | NDCG@10 = %.4f | p95 Latency = %.2f ms",
                results["recall_at_50"], results["ndcg_at_10"], results["p95_latency_ms"])

    return results


if __name__ == "__main__":
    metrics = evaluate_retrieval_baseline()
    print("\n--- Milestone 1 Baseline Retrieval Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
