"""평가 프로토콜 E1~E4 — H2(평가 후보군에 따라 지표가 얼마나 달라지나)의 실험 장치.

  E1 Global-sampled : 정답 + 전역 랜덤 negative n개 (SASRec 원 논문 방식)
  E2 Geo-sampled    : 정답 + 현재 위치 반경 R 안 랜덤 negative n개 (부족하면 가까운 순으로 채움). 정답이 반경 밖이면 miss
  E3 Full           : 전체 아이템
  E4 Geo-full       : 반경 R 안 전체 아이템. 정답이 반경 밖이면 miss (주 지표)

즉 "샘플링 vs 전체" × "전역 vs 동네"의 2×2예요. 동네 프로토콜(E2/E4)에서 반경 밖 정답을 miss로 두는 이유:
동네 필터가 있는 서비스에선 노출될 수 없는 장소이고, E2에서 이를 정답으로 인정하면 negative가 전부 가까운
곳이라 "유일하게 먼 후보 = 정답"이라는 지름길이 생겨요 (E1의 "유일하게 가까운 후보 = 정답"의 반대).

공통 규칙
  - E1/E2 negative는 입력 이력과 정답을 제외해요 (원 논문 관례). 재방문 정답에 유리해지는 효과는 PersonalFreq로 드러내요.
  - 현재 장소(입력의 마지막 장소)는 어떤 후보군에도 넣지 않아요. 연속 중복을 제거했으므로 정의상 정답이 될 수 없어요.
  - E1/E2 후보는 시드 고정으로 한 번 뽑아 저장하고, 모든 모델이 같은 후보로 평가돼요.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from dongne_rec.config import city_dir
from dongne_rec.data.geo import GeoIndex, haversine_km
from dongne_rec.data.split import Queries, Split
from dongne_rec.eval.metrics import expected_metrics

PROTOCOLS = ("E1", "E2", "E3", "E4")
PROTOCOL_NAMES = {"E1": "Global-sampled", "E2": "Geo-sampled", "E3": "Full", "E4": "Geo-full"}


class Scorer(Protocol):
    name: str

    def score(self, histories: list[np.ndarray], last_items: np.ndarray) -> np.ndarray:
        """(len(last_items), N+1) 점수. 0번 열(padding)은 무시돼요."""


def sample_candidates(q: Queries, geo: GeoIndex, n_neg: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    all_items = np.arange(1, geo.n_items + 1)
    e1 = np.empty((len(q), n_neg), dtype=np.int64)
    e2 = np.empty((len(q), n_neg), dtype=np.int64)
    fill = np.zeros(len(q), dtype=np.int64)
    for i in range(len(q)):
        excluded = np.union1d(q.histories[i], [q.targets[i]])
        pool = np.setdiff1d(all_items, excluded, assume_unique=True)
        e1[i] = rng.choice(pool, n_neg, replace=False)

        last = q.last_items[i]
        geo_pool = np.setdiff1d(geo.neighbors(last), excluded, assume_unique=True)
        if len(geo_pool) >= n_neg:
            e2[i] = rng.choice(geo_pool, n_neg, replace=False)
        else:
            outside = pool[~np.isin(pool, geo_pool)]
            nearest = outside[np.argsort(haversine_km(geo.lat[last], geo.lon[last], geo.lat[outside], geo.lon[outside]), kind="stable")]
            fill[i] = n_neg - len(geo_pool)
            e2[i] = np.concatenate([geo_pool, nearest[: fill[i]]])
    return {"E1": e1, "E2": e2, "E2_fill": fill}


def candidates_path(cfg: dict, radius_km: float, n_neg: int, seed: int) -> Path:
    return city_dir(cfg, "processed_dir") / f"eval_candidates_R{radius_km:g}_n{n_neg}_s{seed}.npz"


def load_or_sample_candidates(cfg: dict, split: Split, geo: GeoIndex, n_neg: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """{stage: {"E1", "E2", "E2_fill"}}. 저장된 후보가 있으면 그대로 써서 모델 간·실행 간 후보가 같게 해요."""
    path = candidates_path(cfg, geo.radius_km, n_neg, seed)
    if path.exists():
        with np.load(path) as z:
            cands = {s: {k: z[f"{s}_{k}"] for k in ("E1", "E2", "E2_fill")} for s in ("valid", "test")}
        if all(len(cands[s]["E1"]) == len(split.queries(s)) for s in cands):
            return cands
    rng = np.random.default_rng(seed)
    cands = {s: sample_candidates(split.queries(s), geo, n_neg, rng) for s in ("valid", "test")}
    np.savez_compressed(path, **{f"{s}_{k}": v for s, c in cands.items() for k, v in c.items()})
    return cands


def evaluate(
    model: Scorer, q: Queries, geo: GeoIndex, cands: dict[str, np.ndarray], ks: tuple[int, ...] = (10,), batch_size: int = 256
) -> pd.DataFrame:
    """쿼리 × 프로토콜별 지표 (long format: query, protocol, hr@k, ndcg@k, mrr, in_candidates)."""
    frames = []
    for start in range(0, len(q), batch_size):
        idx = np.arange(start, min(start + batch_size, len(q)))
        scores = np.asarray(model.score([q.histories[i] for i in idx], q.last_items[idx]), dtype=np.float64)
        if scores.shape != (len(idx), geo.n_items + 1):
            raise ValueError(f"{model.name}: score shape {scores.shape} != {(len(idx), geo.n_items + 1)}")
        rows = np.arange(len(idx))
        targets, last = q.targets[idx], q.last_items[idx]
        target_score = scores[rows, targets][:, None]
        near = geo.mask(last)

        counts = {}
        for p, has_target in (("E1", np.ones(len(idx), dtype=bool)), ("E2", near[rows, targets])):
            neg = scores[rows[:, None], cands[p][idx]]
            counts[p] = ((neg > target_score).sum(1), (neg == target_score).sum(1), has_target)

        full = np.ones_like(scores, dtype=bool)
        full[:, 0] = False
        full[rows, last] = False
        for p, mask in (("E3", full), ("E4", near & full)):
            has_target = mask[rows, targets]
            n_equal = ((scores == target_score) & mask).sum(1)
            counts[p] = (((scores > target_score) & mask).sum(1), n_equal - has_target, has_target)

        for p, (greater, ties, has_target) in counts.items():
            metrics = expected_metrics(greater, ties, ks)
            frames.append(pd.DataFrame({
                "query": idx,
                "protocol": p,
                **{name: np.where(has_target, v, 0.0) for name, v in metrics.items()},
                "in_candidates": has_target,
            }))
    return pd.concat(frames, ignore_index=True)
