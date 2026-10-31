"""인기도·개인 빈도 기반 베이스라인. 모두 학습 구간(train) 통계만 써요."""
from __future__ import annotations

import numpy as np

from dongne_rec.data.geo import GeoIndex


def item_counts(train: list[np.ndarray], n_items: int) -> np.ndarray:
    return np.bincount(np.concatenate(train), minlength=n_items + 1).astype(np.float64)


class MostPop:
    name = "MostPop"

    def fit(self, train: list[np.ndarray], n_items: int) -> "MostPop":
        self.counts = item_counts(train, n_items)
        return self

    def score(self, histories: list[np.ndarray], last_items: np.ndarray) -> np.ndarray:
        return np.tile(self.counts, (len(last_items), 1))


class GeoPop(MostPop):
    """현재 위치 반경 R 안 장소를 인기순으로 먼저, 반경 밖은 그 뒤에 인기순."""

    name = "GeoPop"

    def __init__(self, geo: GeoIndex):
        self.geo = geo

    def score(self, histories: list[np.ndarray], last_items: np.ndarray) -> np.ndarray:
        return super().score(histories, last_items) + self.geo.mask(last_items) * (self.counts.max() + 1)


class PersonalFreq(MostPop):
    """본인 방문 빈도 → 같으면 최근 방문 → 이력 밖은 전체 인기순."""

    name = "PersonalFreq"

    def score(self, histories: list[np.ndarray], last_items: np.ndarray) -> np.ndarray:
        scores = np.tile(self.counts / (self.counts.max() + 1), (len(last_items), 1))  # 이력 밖: [0, 1)
        for b, h in enumerate(histories):
            items, freq = np.unique(h, return_counts=True)
            _, first_in_reversed = np.unique(h[::-1], return_index=True)
            last_pos = len(h) - first_in_reversed  # 1..len(h), 클수록 최근
            scores[b, items] = freq * 1e6 + last_pos  # 이력 안: ≥ 1e6 (유저 이력 길이 < 1e6)
        return scores
