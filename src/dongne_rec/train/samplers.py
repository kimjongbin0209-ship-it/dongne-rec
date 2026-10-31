"""학습 negative 샘플러 — H1(동네 안에서 negative를 뽑으면 나아지나)의 실험 장치.

학습 위치 t(입력 s_t, 정답 s_{t+1})마다 negative 1개.
  S0 random: 전체 아이템 균등 (원 SASRec)
공통: negative ≠ 정답, 유저 학습 이력에 있으면 1회 재샘플, padding 위치는 0.
"""
from __future__ import annotations

import numpy as np

SAMPLER_IDS = {"random": "S0", "geo": "S1", "mixed": "S2"}


def user_item_matrix(train: list[np.ndarray], n_items: int) -> np.ndarray:
    """(U, N+1) bool: 유저 u의 학습 이력에 아이템 i가 있는지."""
    m = np.zeros((len(train), n_items + 1), dtype=bool)
    for u, s in enumerate(train):
        m[u, s] = True
    return m


class NegativeSampler:
    name = "base"

    def __init__(self, n_items: int, user_items: np.ndarray, rng: np.random.Generator):
        self.n_items, self.user_items, self.rng = n_items, user_items, rng

    def _draw(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, inputs: np.ndarray, positives: np.ndarray, users: np.ndarray) -> np.ndarray:
        """inputs, positives: (B, L), users: (B,) → negatives (B, L)."""
        negs = self._draw(inputs)
        bad = (negs == positives) | self.user_items[users[:, None], negs]
        negs[bad] = self._draw(inputs)[bad]
        negs[positives == 0] = 0
        return negs


class RandomSampler(NegativeSampler):
    name = "random"

    def _draw(self, inputs: np.ndarray) -> np.ndarray:
        return self.rng.integers(1, self.n_items + 1, size=inputs.shape)
