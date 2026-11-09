"""History-aware reciprocal rank fusion over an eligible candidate set."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

RRF_OFFSET = 60


def reciprocal_ranks(scores: np.ndarray) -> np.ndarray:
    """A tied source score gets the mean of its occupied ranks."""
    return 1.0 / (RRF_OFFSET + rankdata(-scores, method="average"))


def fuse_ranks(
    personal: np.ndarray, sequential: np.ndarray, visited: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Blend source ranks with PersonalFreq weights ordered as (visited, unvisited)."""
    weights = np.asarray(weights, dtype=np.float64)
    personal_weight = np.where(visited, weights[..., 0, None], weights[..., 1, None])
    return personal_weight * personal + (1 - personal_weight) * sequential
