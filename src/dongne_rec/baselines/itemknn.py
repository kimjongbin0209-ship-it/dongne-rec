"""아이템 기반 협업 필터링 (순차 가중).

sim(i, j) = 학습 구간 유저-아이템 이진 행렬의 cosine, 아이템마다 상위 topk 이웃만 유지.
score(j) = Σ_{k=1..L} decay^(k-1) · sim(s_{t-k+1}, j)   (가장 최근 장소부터 L개)
include_self=True면 sim(i, i) = 1을 더해서 이력 안 장소(재방문)도 점수를 받아요.
"""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp


def prune_topk(sim: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """행마다 값이 큰 k개만 남겨요."""
    sim = sim.tocsr()
    keep = np.zeros(sim.nnz, dtype=bool)
    for r in range(sim.shape[0]):
        a, b = sim.indptr[r], sim.indptr[r + 1]
        if b - a <= k:
            keep[a:b] = True
        else:
            keep[a + np.argpartition(-sim.data[a:b], k)[:k]] = True
    rows = np.repeat(np.arange(sim.shape[0]), np.diff(sim.indptr))
    return sp.csr_matrix((sim.data[keep], (rows[keep], sim.indices[keep])), shape=sim.shape)


class ItemKNN:
    def __init__(self, topk: int = 100, last_n: int | None = None, decay: float = 1.0, include_self: bool = True):
        self.topk, self.last_n, self.decay, self.include_self = topk, last_n, decay, include_self

    @property
    def name(self) -> str:
        return f"ItemKNN(L={self.last_n or 'all'}, decay={self.decay:g}, self={self.include_self})"

    def fit(self, train: list[np.ndarray], n_items: int) -> "ItemKNN":
        rows = np.repeat(np.arange(len(train)), [len(s) for s in train])
        x = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, np.concatenate(train))), shape=(len(train), n_items + 1))
        x.data[:] = 1.0  # 방문 횟수가 아닌 방문 여부
        co = (x.T @ x).tocsr()
        inv_norm = 1.0 / np.sqrt(np.maximum(co.diagonal(), 1.0))
        sim = sp.diags(inv_norm) @ co @ sp.diags(inv_norm)
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.sim = prune_topk(sim, self.topk)
        return self

    def configure(self, **params) -> "ItemKNN":
        """학습된 유사도를 공유하는 사본 (튜닝용)."""
        other = copy.copy(self)
        for k, v in params.items():
            setattr(other, k, v)
        return other

    def score(self, histories: list[np.ndarray], last_items: np.ndarray) -> np.ndarray:
        rows, cols, vals = [], [], []
        for b, h in enumerate(histories):
            recent = h[::-1] if self.last_n is None else h[::-1][: self.last_n]
            rows.append(np.full(len(recent), b))
            cols.append(recent)
            vals.append(self.decay ** np.arange(len(recent), dtype=np.float64))
        w = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(histories), self.sim.shape[0]))
        scores = (w @ self.sim).toarray()
        if self.include_self:
            scores += w.toarray()
        return scores
