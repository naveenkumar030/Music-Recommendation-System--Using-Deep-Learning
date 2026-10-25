"""
Deep Learning Recommendation Training Pipeline.

Trains the Multi-Task Deep Neural Ranker:
1. Loads processed users, songs, and historical listening interaction sequences.
2. Formulates multi-objective training targets (P(completion), P(like), P(skip)).
3. Trains MultiTaskDeepRanker with AdamW optimizer and Cosine Annealing learning rate.
4. Evaluates offline Top-K ranking performance (HR@5, HR@10, HR@20, NDCG@10, MRR, Gini diversity).
5. Exports production checkpoint to `models/deep_ranker.pt`.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.feature_store import FeatureStore
from src.models.retrieval import TwoTowerModel, VectorIndex
from src.models.deep_ranker import MultiTaskDeepRanker, DeepRankerAgent
from src.eval.dl_evaluation import evaluate_deep_ranking_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DeepRankingDataset(Dataset):
    """
    Dataset for Deep Multi-Task Ranking:
    Yields (state_vector, candidate_song_embedding, completion_target, like_target, skip_target)
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        feature_store: FeatureStore,
        song_embeddings: np.ndarray,
        history_window: int = 5
    ):
        self.feature_store = feature_store
        self.song_embeddings = song_embeddings
        self.history_window = history_window
        self.samples = []

        grouped = events_df.groupby("user_id")

        for user_id, group in grouped:
            if user_id not in feature_store.user_id_to_idx:
                continue
            u_vec = feature_store.get_user_vector_by_id(user_id)
            records = group.sort_values("timestamp").to_dict("records") if "timestamp" in group.columns else group.to_dict("records")

            history_events = []
            for row in records:
                sid = row.get("song_id")
                if sid not in feature_store.song_id_to_idx:
                    continue

                s_idx = feature_store.song_id_to_idx[sid]
                song_vec = self.song_embeddings[s_idx] if s_idx < len(self.song_embeddings) else np.zeros(self.song_embeddings.shape[1], dtype=np.float32)

                skip_t = row.get("skip_type", "no_skip")
                is_like = float(row.get("liked", 0))

                t_complete = 1.0 if skip_t == "no_skip" else 0.0
                t_like = 1.0 if is_like == 1 else 0.0
                t_skip = 1.0 if skip_t in ["skip_early", "skip_late"] else 0.0

                # Build flat history vector
                hist_feats = []
                pad_count = self.history_window - len(history_events[-self.history_window:])
                for _ in range(pad_count):
                    hist_feats.extend([0.0] * (feature_store.song_dim + 3))
                for h_sid, h_skip, h_lk in history_events[-self.history_window:]:
                    h_s_vec = feature_store.get_song_vector_by_id(h_sid)
                    hist_feats.extend(list(h_s_vec) + [
                        1.0 if h_skip == "no_skip" else 0.0,
                        1.0 if h_skip == "skip_early" else 0.0,
                        float(h_lk)
                    ])

                ctx_vec = [14.0 / 24.0, 3.0 / 7.0, 1.0]
                skips_so_far = sum(1 for _, s, _ in history_events if s != "no_skip")
                likes_so_far = sum(1 for _, _, l in history_events if l == 1)
                tot = max(1, len(history_events))
                stats_vec = [skips_so_far / tot, likes_so_far / 10.0, tot / 50.0]

                state_vec = np.concatenate([
                    u_vec,
                    np.array(hist_feats, dtype=np.float32),
                    np.array(ctx_vec, dtype=np.float32),
                    np.array(stats_vec, dtype=np.float32)
                ]).astype(np.float32)

                self.samples.append((
                    state_vec,
                    song_vec.astype(np.float32),
                    np.float32(t_complete),
                    np.float32(t_like),
                    np.float32(t_skip)
                ))

                history_events.append((sid, skip_t, is_like))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_vec, song_vec, t_comp, t_like, t_skip = self.samples[idx]
        return (
            torch.tensor(s_vec, dtype=torch.float32),
            torch.tensor(song_vec, dtype=torch.float32),
            torch.tensor(t_comp, dtype=torch.float32),
            torch.tensor(t_like, dtype=torch.float32),
            torch.tensor(t_skip, dtype=torch.float32)
        )


def train_deep_ranker(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports",
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-3
) -> Dict[str, Any]:
    """Executes the full Deep Learning ranking training workflow."""
    p_dir = Path(processed_dir)
    m_dir = Path(models_dir)
    r_dir = Path(reports_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing FeatureStore and loading parquet datasets...")
    users_df = pd.read_parquet(p_dir / "users.parquet")
    songs_df = pd.read_parquet(p_dir / "songs.parquet")
    train_events = pd.read_parquet(p_dir / "train_events.parquet")
    val_events = pd.read_parquet(p_dir / "val_events.parquet")

    fs = FeatureStore(processed_dir=processed_dir)
    fs.load_and_index(users_df, songs_df)

    # Load pre-trained Two-Tower model & song embeddings
    two_tower_path = m_dir / "baseline_two_tower.pt"
    song_embs_path = m_dir / "song_embeddings.npy"

    retrieval_model = TwoTowerModel(user_dim=fs.user_dim, song_dim=fs.song_dim, embed_dim=64)
    if two_tower_path.exists():
        retrieval_model.load_state_dict(torch.load(two_tower_path, weights_only=True))
        logger.info("Loaded pre-trained Two-Tower retrieval model.")

    if song_embs_path.exists():
        song_embeddings = np.load(song_embs_path)
        logger.info(f"Loaded song embeddings matrix from {song_embs_path} with shape {song_embeddings.shape}")
    else:
        with torch.no_grad():
            song_embeddings = retrieval_model.song_tower(fs.song_tensor).cpu().numpy()
        np.save(song_embs_path, song_embeddings)

    history_window = 5
    state_dim = fs.user_dim + history_window * (fs.song_dim + 3) + 3 + 3
    song_embed_dim = song_embeddings.shape[1]

    logger.info(f"Deep Ranker Config: State Dim = {state_dim}, Song Embedding Dim = {song_embed_dim}")

    train_ds = DeepRankingDataset(train_events, fs, song_embeddings=song_embeddings, history_window=history_window)
    val_ds = DeepRankingDataset(val_events, fs, song_embeddings=song_embeddings, history_window=history_window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTaskDeepRanker(state_dim=state_dim, song_dim=song_embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_loss = nn.BCELoss()

    logger.info(f"Beginning Deep Neural Ranker training over {epochs} epochs ({len(train_ds)} samples)...")
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_comp_loss = 0.0
        total_like_loss = 0.0
        total_skip_loss = 0.0

        for s_b, song_b, comp_b, like_b, skip_b in train_loader:
            optimizer.zero_grad()
            out = model(s_b, song_b)

            l_comp = bce_loss(out["p_complete"].squeeze(-1), comp_b)
            l_like = bce_loss(out["p_like"].squeeze(-1), like_b)
            l_skip = bce_loss(out["p_skip"].squeeze(-1), skip_b)

            loss = l_comp + 1.5 * l_like + 1.0 * l_skip
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_comp_loss += l_comp.item()
            total_like_loss += l_like.item()
            total_skip_loss += l_skip.item()

        scheduler.step()
        n_batches = max(1, len(train_loader))
        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(total_loss / n_batches, 4),
            "completion_loss": round(total_comp_loss / n_batches, 4),
            "like_loss": round(total_like_loss / n_batches, 4),
            "skip_loss": round(total_skip_loss / n_batches, 4),
        }
        training_history.append(epoch_summary)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_summary['train_loss']:.4f} (Comp: {epoch_summary['completion_loss']:.4f}, Like: {epoch_summary['like_loss']:.4f}, Skip: {epoch_summary['skip_loss']:.4f})")

    # Save trained checkpoint
    save_path = m_dir / "deep_ranker.pt"
    torch.save(model.state_dict(), str(save_path))
    logger.info(f"✅ Saved Deep Neural Ranker checkpoint to {save_path}")

    # Build Agent and evaluate offline ranking metrics
    agent = DeepRankerAgent(state_dim=state_dim, song_embeddings=song_embeddings, feature_store=fs)
    agent.ranker = model

    logger.info("Evaluating offline Deep Learning ranking metrics (HR@K, NDCG@K, MRR)...")
    eval_metrics = evaluate_deep_ranking_model(
        feature_store=fs,
        retrieval_model=retrieval_model,
        deep_ranker=agent,
        val_events_df=val_events,
        k_list=[5, 10, 20]
    )

    final_report = {
        "model_architecture": "Two-Tower DSSM Retrieval + Multi-Task Deep Neural Ranker",
        "training_epochs": epochs,
        "final_train_loss": training_history[-1]["train_loss"],
        "training_history": training_history,
        "ranking_evaluation": eval_metrics
    }

    report_path = r_dir / "deep_learning_training_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"✅ Deep Learning training report saved to {report_path}")

    print("\n" + "=" * 60)
    print("DEEP LEARNING MUSIC RECOMMENDATION SYSTEM — TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Loss:    {training_history[-1]['train_loss']}")
    print(f"Hit Rate @ 10: {eval_metrics.get('HR@10', 0.0) * 100:.2f}%")
    print(f"NDCG @ 10:     {eval_metrics.get('NDCG@10', 0.0):.4f}")
    print(f"MRR:           {eval_metrics.get('MRR', 0.0):.4f}")
    print(f"Coverage:      {eval_metrics.get('catalog_coverage', 0.0) * 100:.2f}%")
    print(f"Gini Index:    {eval_metrics.get('gini_diversity', 0.0):.4f}")
    print("=" * 60 + "\n")

    return final_report


if __name__ == "__main__":
    train_deep_ranker()
