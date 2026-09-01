"""
Unit and Integration Tests for Milestone 4: OPE and Guardrails.
"""

import numpy as np
import pytest

from src.eval.ope import compute_ips_and_doubly_robust
from src.eval.guardrails import compute_gini_coefficient, compute_guardrail_metrics


def test_gini_coefficient():
    # Perfectly uniform distribution: items = ['a', 'b', 'c', 'd'] -> Gini should be low
    uniform_items = ["a", "b", "c", "d"]
    gini_uni = compute_gini_coefficient(uniform_items)
    assert 0.0 <= gini_uni < 0.1

    # Concentrated distribution: 95% 'a', 5% 'b' -> Gini should be high
    skewed_items = ["a"] * 95 + ["b"] * 5
    gini_skew = compute_gini_coefficient(skewed_items)
    assert gini_skew > 0.4


def test_guardrail_metrics():
    song_ids = ["s1", "s2", "s3", "s1"]
    artists = ["Art1", "Art2", "Art3", "Art1"]
    genres = ["Pop", "Rock", "Pop", "Jazz"]

    metrics = compute_guardrail_metrics(
        recommended_song_ids=song_ids,
        recommended_artists=artists,
        recommended_genres=genres,
        total_catalog_songs=10,
        total_catalog_artists=5
    )

    assert "catalog_coverage_pct" in metrics
    assert "artist_gini_coefficient" in metrics
    assert metrics["catalog_coverage_pct"] == 30.0  # 3 unique out of 10


def test_ope_estimators():
    n = 100
    logged_events = [{"id": i} for i in range(n)]
    pi_probs = np.random.uniform(0.01, 0.1, size=n)
    mu_probs = np.random.uniform(0.01, 0.1, size=n)
    rewards = np.random.normal(1.0, 0.5, size=n)
    q_vals = np.random.normal(1.0, 0.3, size=n)
    v_vals = np.random.normal(1.0, 0.2, size=n)

    results = compute_ips_and_doubly_robust(
        logged_events=logged_events,
        target_policy_probs=pi_probs,
        logging_policy_probs=mu_probs,
        rewards=rewards,
        estimated_q_values=q_vals,
        estimated_v_values=v_vals,
        clip_max=5.0
    )

    assert "v_ips" in results
    assert "v_dr" in results
    assert "effective_sample_size" in results
    assert not np.isnan(results["v_dr"])
