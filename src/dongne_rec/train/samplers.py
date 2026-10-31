"""학습 negative 샘플러 — H1(동네 안에서 negative를 뽑으면 나아지나)의 실험 장치.

학습 위치 t(입력 s_t, 정답 s_{t+1})마다 negative 1개.
  S0 random: 전체 아이템 균등 (원 SASRec)
  S1 geo   : 현재 위치 loc(s_t) 반경 R 안에서 균등
  S2 mixed : 위치마다 확률 α로 S1, 1−α로 S0
공통: negative ≠ 정답, 유저 학습 이력에 있으면 1회 재샘플, padding 위치는 0.
S1/S2는 반경이 좁아 재샘플 뒤에도 정답·이력과 겹치는 위치를 전역 랜덤으로 대체해요 (겹치지 않을 때까지 최대 3회).
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

    def _collides(self, negs: np.ndarray, positives: np.ndarray, users: np.ndarray) -> np.ndarray:
        return (negs == positives) | self.user_items[users[:, None], negs]

    def sample(self, inputs: np.ndarray, positives: np.ndarray, users: np.ndarray) -> np.ndarray:
        """inputs, positives: (B, L), users: (B,) → negatives (B, L)."""
        negs = self._draw(inputs)
        bad = self._collides(negs, positives, users)
        negs[bad] = self._draw(inputs)[bad]
        negs[positives == 0] = 0
        return negs

    def _random(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.integers(1, self.n_items + 1, size=shape)


class RandomSampler(NegativeSampler):
    name = "random"

    def _draw(self, inputs: np.ndarray) -> np.ndarray:
        return self._random(inputs.shape)


class GeoSampler(NegativeSampler):
    name = "geo"

    def __init__(self, n_items: int, user_items: np.ndarray, rng: np.random.Generator, indptr: np.ndarray, indices: np.ndarray):
        super().__init__(n_items, user_items, rng)
        self.indptr, self.indices = indptr, indices

    def _geo(self, inputs: np.ndarray) -> np.ndarray:
        """입력 장소 반경 안에서 균등 추출 (이웃 CSR에서 벡터화). 반경 안이 비면(padding) 전역 랜덤."""
        start = self.indptr[inputs]
        deg = self.indptr[inputs + 1] - start
        pick = self.indices[np.minimum(start + (self.rng.random(inputs.shape) * deg).astype(np.int64), len(self.indices) - 1)]
        return np.where(deg > 0, pick, self._random(inputs.shape))

    def _draw(self, inputs: np.ndarray) -> np.ndarray:
        return self._geo(inputs)

    def sample(self, inputs: np.ndarray, positives: np.ndarray, users: np.ndarray, fallback_rounds: int = 3) -> np.ndarray:
        negs = super().sample(inputs, positives, users)
        for _ in range(fallback_rounds):
            still_bad = self._collides(negs, positives, users) & (positives > 0)
            if not still_bad.any():
                break
            negs[still_bad] = self._random(inputs.shape)[still_bad]
        return negs


class MixedSampler(GeoSampler):
    name = "mixed"

    def __init__(self, *args, geo_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.geo_ratio = geo_ratio

    def _draw(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(self.rng.random(inputs.shape) < self.geo_ratio, self._geo(inputs), self._random(inputs.shape))
