"""정답이 1개인 랭킹 지표.

정답보다 점수가 높은 후보 수 g, 정답과 점수가 같은 다른 후보 수 t가 주어지면 정답 순위는
g+1 .. g+t+1 중 하나예요. 동점을 무작위로 섞는다고 보고 그 기댓값을 계산해서
동점이 많은 인기순 계열에서도 낙관/비관 편향이나 시드 노이즈가 없어요.
"""
from __future__ import annotations

import numpy as np


def _discount_cumsum(k: int) -> np.ndarray:
    """c[r] = Σ_{j=1..r} 1/log2(j+1), c[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])


def _harmonic(n: int) -> np.ndarray:
    """h[r] = Σ_{j=1..r} 1/j, h[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])


def expected_metrics(n_greater, n_ties, ks: tuple[int, ...] = (10,)) -> dict[str, np.ndarray]:
    g = np.asarray(n_greater, dtype=np.int64)
    t = np.asarray(n_ties, dtype=np.int64)
    out = {}
    for k in ks:
        lo, hi = np.minimum(g, k), np.minimum(g + t + 1, k)
        disc = _discount_cumsum(k)
        out[f"hr@{k}"] = (hi - lo) / (t + 1)
        out[f"ndcg@{k}"] = (disc[hi] - disc[lo]) / (t + 1)
    h = _harmonic(int((g + t + 1).max(initial=1)))
    out["mrr"] = (h[g + t + 1] - h[g]) / (t + 1)
    return out
