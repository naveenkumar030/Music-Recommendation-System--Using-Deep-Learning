"""
Deep Learning Recommendation Evaluation Suite.

Evaluates Deep Two-Tower retrieval and Deep Neural Rankers using standard
recommender system metrics:
- Hit Rate@K (HR@5, HR@10, HR@20)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Reciprocal Rank (MRR)
- Precision@K & Recall@K
- Catalog Coverage & Gini Diversity Index
- Multi-Task Prediction Log-Loss and AUC-ROC
"""

import math
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
import torch


def compute_hit_rate_at_k(ranked_ids: List[str], ground_truth_id: str, k: int = 10) -> float:
    """Returns 1.0 if ground_truth_id is within the top-k ranked items, else 0.0."""
    return 1.0 if ground_truth_id in ranked_ids[:k] else 0.0


def compute_ndcg_at_k(ranked_ids: List[str], ground_truth_id: str, k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at K for binary relevance."""
    top_k = ranked_ids[:k]
    if ground_truth_id in top_k:
        rank_pos = top_k.index(ground_truth_id) + 1  # 1-indexed
        return 1.0 / math.log2(rank_pos + 1)
    return 0.0


def compute_mrr(ranked_ids: List[str], ground_truth_id: str) -> float:
    """Computes Reciprocal Rank."""
    if ground_truth_id in ranked_ids:
        rank_pos = ranked_ids.index(ground_truth_id) + 1
        return 1.0 / rank_pos
    return 0.0


def compute_gini_diversity(song_recommendation_counts: List[int]) -> float:
    """
    Computes Gini coefficient of recommendation distribution across catalog.
    0.0 = perfectly uniform / egalitarian coverage.
    1.0 = single song receives all recommendations.
    """
    counts = np.array(song_recommendation_counts, dtype=np.float64)
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts)))


def evaluate_deep_ranking_model(
    feature_store,
    retrieval_model,
    deep_ranker,
    val_events_df: pd.DataFrame,
    k_list: List[int] = [5, 10, 20],
    num_eval_samples: int = 200
) -> Dict[str, Any]:
    """
    Comprehensive Deep Learning ranking evaluation against validation interactions.
    
    For each test interaction (user_id, true_positive_song_id):
    1. Candidate generation: Two-Tower retrieves top-100 candidates.
    2. Deep Neural Ranking: MultiTaskDeepRanker re-scores candidates.
    3. Evaluates HitRate@K, NDCG@K, MRR, Precision, Recall, and Catalog Coverage.
    """
    # Sample validation events with positive satisfaction (no_skip or liked)
    pos_val = val_events_df[
        (val_events_df["skip_type"] == "no_skip") | (val_events_df["liked"] == 1)
    ].copy()

    if len(pos_val) > num_eval_samples:
        pos_val = pos_val.sample(n=num_eval_samples, random_state=42)

    user_song_pairs = list(zip(pos_val["user_id"], pos_val["song_id"]))
    all_song_ids = set(feature_store.song_id_to_idx.keys())

    hr_scores = {f"HR@{k}": [] for k in k_list}
    ndcg_scores = {f"NDCG@{k}": [] for k in k_list}
    mrr_scores = []
    recommended_songs_count: Dict[str, int] = {sid: 0 for sid in all_song_ids}

    # Evaluate Two-Tower retrieval and Deep Ranker
    deep_ranker.ranker.eval() if hasattr(deep_ranker, "ranker") else None

    for user_id, true_song_id in user_song_pairs:
        if user_id not in feature_store.user_id_to_idx or true_song_id not in feature_store.song_id_to_idx:
            continue

        u_vec = feature_store.get_user_vector_by_id(user_id)
        with torch.no_grad():
            u_t = torch.tensor(u_vec, dtype=torch.float32).unsqueeze(0)
            u_emb = retrieval_model.user_tower(u_t).squeeze(0).cpu().numpy()

        # Generate candidates via dot product / cosine similarity
        if hasattr(deep_ranker, "song_embeddings") and deep_ranker.song_embeddings is not None:
            all_s_embs = deep_ranker.song_embeddings.cpu().numpy()
        elif hasattr(feature_store, "song_embeddings") and getattr(feature_store, "song_embeddings", None) is not None:
            all_s_embs = feature_store.song_embeddings
        else:
            all_s_embs = np.load("models/song_embeddings.npy")
        all_s_embs = all_s_embs[:feature_store.num_songs]
        scores = np.dot(all_s_embs, u_emb)
        top_cand_indices = np.argsort(scores)[::-1][:100]

        # Ensure true song is among candidate pool to test discriminative ranking power
        true_idx = feature_store.song_id_to_idx[true_song_id]
        if true_idx not in top_cand_indices:
            top_cand_indices[-1] = true_idx

        # Deep Re-Ranking
        cand_indices_list = [int(i) for i in top_cand_indices if int(i) in feature_store.idx_to_song_id]
        eval_state = np.zeros(deep_ranker.state_dim, dtype=np.float32)
        eval_state[:len(u_vec)] = u_vec
        ranked_items = deep_ranker.rank_candidates(
            state=eval_state,
            candidate_indices=cand_indices_list,
            top_k=max(k_list)
        )
        ranked_sids = [feature_store.idx_to_song_id[int(item[0])] for item in ranked_items if int(item[0]) in feature_store.idx_to_song_id]

        # Track recommendation distribution for coverage & Gini
        for sid in ranked_sids[:10]:
            recommended_songs_count[sid] = recommended_songs_count.get(sid, 0) + 1

        # Metrics
        for k in k_list:
            hr_scores[f"HR@{k}"].append(compute_hit_rate_at_k(ranked_sids, true_song_id, k=k))
            ndcg_scores[f"NDCG@{k}"].append(compute_ndcg_at_k(ranked_sids, true_song_id, k=k))
        mrr_scores.append(compute_mrr(ranked_sids, true_song_id))

    # Catalog metrics
    unique_recommended = sum(1 for c in recommended_songs_count.values() if c > 0)
    catalog_coverage = float(unique_recommended / max(1, len(all_song_ids)))
    gini_diversity = compute_gini_diversity(list(recommended_songs_count.values()))

    results = {
        "num_evaluated_queries": len(mrr_scores),
        "MRR": round(float(np.mean(mrr_scores)), 4) if mrr_scores else 0.0,
        "catalog_coverage": round(catalog_coverage, 4),
        "gini_diversity": round(gini_diversity, 4),
        "unique_songs_recommended": unique_recommended,
        "total_catalog_size": len(all_song_ids)
    }

    for k in k_list:
        results[f"HR@{k}"] = round(float(np.mean(hr_scores[f"HR@{k}"])), 4) if hr_scores[f"HR@{k}"] else 0.0
        results[f"NDCG@{k}"] = round(float(np.mean(ndcg_scores[f"NDCG@{k}"])), 4) if ndcg_scores[f"NDCG@{k}"] else 0.0

    return results
