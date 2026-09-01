"""
Off-Policy Evaluation (OPE) Module for Recommendation Policies.

Implements:
- Inverse Propensity Scoring (IPS) / Importance Sampling with weight clipping.
- Direct Method (DM) using learned Q-function.
- Doubly-Robust (DR) estimator combining IPS and DM for unbiased, low-variance estimation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


def compute_ips_and_doubly_robust(
    logged_events: List[Dict],
    target_policy_probs: np.ndarray,      # (N_events,) probability target policy took logged action
    logging_policy_probs: np.ndarray,     # (N_events,) propensity score P(a | s) in log
    rewards: np.ndarray,                  # (N_events,) observed reward
    estimated_q_values: np.ndarray,       # (N_events,) Critic predicted Q(s, a_logged)
    estimated_v_values: np.ndarray,       # (N_events,) Critic predicted V(s) = sum_a pi(a|s) Q(s, a)
    clip_max: float = 10.0
) -> Dict[str, float]:
    """
    Computes IPS, Direct Method (DM), and Doubly-Robust (DR) policy value estimates.
    """
    assert len(logged_events) == len(target_policy_probs) == len(logging_policy_probs) == len(rewards)

    # 1. Importance Weights w = pi(a|s) / mu(a|s)
    weights = target_policy_probs / np.clip(logging_policy_probs, 1e-4, 1.0)
    clipped_weights = np.clip(weights, 0.0, clip_max)

    # 2. Importance Sampling (IPS)
    # V_IPS = (1/N) * sum_i w_i * r_i
    ips_values = clipped_weights * rewards
    v_ips = float(np.mean(ips_values))
    ips_std = float(np.std(ips_values) / np.sqrt(len(ips_values)))

    # 3. Direct Method (DM)
    # V_DM = (1/N) * sum_i V_hat(s_i)
    v_dm = float(np.mean(estimated_v_values))

    # 4. Doubly-Robust (DR) Estimator
    # V_DR = (1/N) * sum_i [ V_hat(s_i) + w_i * (r_i - Q_hat(s_i, a_i)) ]
    dr_values = estimated_v_values + clipped_weights * (rewards - estimated_q_values)
    v_dr = float(np.mean(dr_values))
    dr_std = float(np.std(dr_values) / np.sqrt(len(dr_values)))

    # Effective Sample Size (ESS) = (sum w)^2 / sum(w^2)
    sum_w = np.sum(clipped_weights)
    sum_w_sq = np.sum(clipped_weights ** 2)
    ess = float((sum_w ** 2) / max(1e-6, sum_w_sq))

    return {
        "v_ips": round(v_ips, 4),
        "ips_std": round(ips_std, 4),
        "v_dm": round(v_dm, 4),
        "v_dr": round(v_dr, 4),
        "dr_std": round(dr_std, 4),
        "mean_importance_weight": round(float(np.mean(clipped_weights)), 4),
        "max_importance_weight": round(float(np.max(clipped_weights)), 4),
        "effective_sample_size": round(ess, 1),
        "num_evaluated_events": len(logged_events)
    }
