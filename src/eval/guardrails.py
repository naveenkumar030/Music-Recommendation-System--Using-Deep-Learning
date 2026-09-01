"""
Guardrails and Diversity Analysis Module.

Computes:
- Gini Coefficient (artist and genre concentration).
- Catalog Coverage (% of catalog recommended across evaluation cohort).
- Intra-List Diversity and Novelty Rate.
- Reward Ablation studies.
"""

from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


def compute_gini_coefficient(items: List[str]) -> float:
    """
    Computes Gini inequality coefficient for item distribution (0 = perfectly uniform, 1 = concentrated on one item).
    """
    if not items:
        return 0.0
    counts = np.array(list(Counter(items).values()), dtype=np.float64)
    counts.sort()
    n = len(counts)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts)))


def compute_guardrail_metrics(
    recommended_song_ids: List[str],
    recommended_artists: List[str],
    recommended_genres: List[str],
    total_catalog_songs: int,
    total_catalog_artists: int
) -> Dict[str, float]:
    """
    Computes catalog coverage, artist concentration, and genre entropy.
    """
    unique_songs = len(set(recommended_song_ids))
    unique_artists = len(set(recommended_artists))
    
    catalog_coverage = unique_songs / max(1, total_catalog_songs)
    artist_coverage = unique_artists / max(1, total_catalog_artists)
    
    artist_gini = compute_gini_coefficient(recommended_artists)
    genre_gini = compute_gini_coefficient(recommended_genres)

    # Genre Shannon entropy
    genre_counts = np.array(list(Counter(recommended_genres).values()), dtype=np.float64)
    genre_probs = genre_counts / np.sum(genre_counts)
    genre_entropy = -np.sum(genre_probs * np.log2(genre_probs + 1e-12))

    return {
        "catalog_coverage_pct": round(catalog_coverage * 100.0, 2),
        "artist_coverage_pct": round(artist_coverage * 100.0, 2),
        "artist_gini_coefficient": round(artist_gini, 4),
        "genre_gini_coefficient": round(genre_gini, 4),
        "genre_shannon_entropy": round(float(genre_entropy), 3),
        "total_recommendations": len(recommended_song_ids)
    }
