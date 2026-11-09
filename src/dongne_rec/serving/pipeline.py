"""동네 후보를 SASRec으로 재랭킹합니다.

기본 정책은 고정 반경 안의 전체 후보를 평가합니다. 확장 모드는 후보가 k개 미만일 때
반경을 지정한 횟수까지 두 배로 늘립니다. FAISS와 NumPy 후보 축소는 별도 옵션입니다.
현재 장소는 오프라인 E4와 동일하게 제외합니다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dongne_rec.data.geo import EARTH_RADIUS_KM, build_ball_tree, item_coords
from dongne_rec.data.sequences import pad_left

DEFAULT_SERVING_CONFIG = "configs/serving.yaml"
DEFAULT_PIPELINE = {
    "retrieval": "full",
    "radius_mode": "fixed",
    "n_candidates": 200,
    "query_last_n": 3,
    "max_radius_expansions": 3,
}


def pipeline_settings(config: dict) -> dict:
    return {**DEFAULT_PIPELINE, **config.get("pipeline", {})}


def artifacts_dir(scfg: dict, city: str, radius: float, sampler: str, seed: int) -> Path:
    return Path(scfg["artifacts_dir"]) / city / f"R{radius:g}" / f"{sampler}_s{seed}"


class OnnxEncoder:
    def __init__(self, path: Path, threads: int = 0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 = ORT 기본
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.name = f"onnx-{Path(path).stem.split('_')[-1]}"

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"seq": seq})[0]


_TORCH_DEFAULT_THREADS: int | None = None


class TorchEncoder:
    def __init__(self, model, device: str = "cpu", threads: int = 0):
        import torch

        global _TORCH_DEFAULT_THREADS
        if _TORCH_DEFAULT_THREADS is None:  # 한 번 1로 바꾼 뒤에도 '기본'으로 되돌릴 수 있게 처음 값을 기억
            _TORCH_DEFAULT_THREADS = torch.get_num_threads()
        torch.set_num_threads(threads or _TORCH_DEFAULT_THREADS)
        self.torch, self.device = torch, torch.device(device)
        self.model = model.to(self.device).eval()
        self.name = f"torch-{device}"

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(seq).to(self.device)).cpu().numpy()


@dataclass
class Recommendation:
    items: np.ndarray
    scores: np.ndarray
    n_geo: int
    n_candidates: int
    radius_km: float
    candidates: np.ndarray = field(repr=False)
    candidate_scores: np.ndarray = field(repr=False)
    timings_ms: dict[str, float] = field(default_factory=dict)


class TwoStageRecommender:
    def __init__(
        self, item_emb: np.ndarray, lat: np.ndarray, lon: np.ndarray, encoder, max_len: int, radius_km: float,
        n_candidates: int = DEFAULT_PIPELINE["n_candidates"], query_last_n: int = DEFAULT_PIPELINE["query_last_n"],
        max_radius_expansions: int = DEFAULT_PIPELINE["max_radius_expansions"],
        retrieval: str = DEFAULT_PIPELINE["retrieval"], radius_mode: str = DEFAULT_PIPELINE["radius_mode"],
    ):
        if retrieval not in ("full", "faiss", "numpy"):
            raise ValueError("retrieval은 full, faiss, numpy 중 하나여야 합니다.")
        if radius_mode not in ("fixed", "expand"):
            raise ValueError("radius_mode는 fixed 또는 expand여야 합니다.")
        for name, value, minimum in (
            ("n_candidates", n_candidates, 1),
            ("query_last_n", query_last_n, 1),
            ("max_radius_expansions", max_radius_expansions, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}은 {minimum} 이상의 정수여야 합니다.")
        self.item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        self.lat, self.lon = lat, lon
        self.encoder, self.max_len, self.radius_km = encoder, max_len, radius_km
        self.n_candidates, self.query_last_n, self.max_radius_expansions = n_candidates, query_last_n, max_radius_expansions
        self.retrieval, self.radius_mode = retrieval, radius_mode
        self.tree = build_ball_tree(lat, lon)
        if retrieval == "faiss":
            import faiss  # macOS의 Torch OpenMP 런타임과 분리합니다.

            self.faiss = faiss
            self.index = faiss.IndexFlatIP(self.item_emb.shape[1])
            self.index.add(self.item_emb[1:])  # FAISS id p ↔ item_idx p+1

    @property
    def n_items(self) -> int:
        return len(self.item_emb) - 1

    def geo_filter(self, lat: float, lon: float, exclude: int, k: int, radius_km: float) -> tuple[np.ndarray, float]:
        query = np.radians([[lat, lon]])
        remaining = self.max_radius_expansions if self.radius_mode == "expand" else 0
        while True:
            items = self.tree.query_radius(query, r=radius_km / EARTH_RADIUS_KM)[0] + 1
            items = items[items != exclude]
            if len(items) >= k or remaining == 0:
                return items, radius_km
            remaining -= 1
            radius_km *= 2

    def narrow(self, recent: np.ndarray, geo_items: np.ndarray) -> np.ndarray:
        if self.retrieval == "full" or len(geo_items) <= self.n_candidates:
            return geo_items
        query = self.item_emb[recent[-self.query_last_n :]].mean(axis=0, keepdims=True)
        if self.retrieval == "faiss":
            params = self.faiss.SearchParameters(sel=self.faiss.IDSelectorBatch(geo_items - 1))
            _, ids = self.index.search(query, self.n_candidates, params=params)
            return ids[0][ids[0] >= 0] + 1
        scores = self.item_emb[geo_items] @ query[0]
        return geo_items[np.argpartition(-scores, self.n_candidates)[: self.n_candidates]]

    def _rank(self, recent: np.ndarray, candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not len(candidates):
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float32)
        h = self.encoder(pad_left([recent], self.max_len))[0]
        scores = self.item_emb[candidates] @ h
        return np.argsort(-scores, kind="stable")[:k], scores

    def rerank(self, recent: np.ndarray, candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        top, scores = self._rank(recent, candidates, k)
        return candidates[top], scores[top]

    def recommend(self, recent: np.ndarray, lat: float, lon: float, k: int = 10, radius_km: float | None = None) -> Recommendation:
        recent = np.asarray(recent, dtype=np.int64)[-self.max_len :]
        t0 = time.perf_counter_ns()
        base_radius = self.radius_km if radius_km is None else radius_km
        geo_items, radius_used = self.geo_filter(lat, lon, int(recent[-1]), k, base_radius)
        t1 = time.perf_counter_ns()
        candidates = self.narrow(recent, geo_items)
        t2 = time.perf_counter_ns()
        top, candidate_scores = self._rank(recent, candidates, k)
        items, scores = candidates[top], candidate_scores[top]
        t3 = time.perf_counter_ns()
        return Recommendation(
            items, scores, len(geo_items), len(candidates), radius_used, candidates, candidate_scores,
            {"geo_filter": (t1 - t0) / 1e6, "narrow": (t2 - t1) / 1e6, "rerank": (t3 - t2) / 1e6, "total": (t3 - t0) / 1e6},
        )


def load_recommender(artifacts: Path, backend: str = "onnx-int8", threads: int = 0, **pipeline_kwargs) -> tuple[TwoStageRecommender, pd.DataFrame, dict]:
    """backend: onnx-fp32 | onnx-int8 | torch-cpu | torch-mps. (추천기, items, meta)를 돌려줘요."""
    if backend.startswith("torch-") and pipeline_kwargs.get("retrieval", DEFAULT_PIPELINE["retrieval"]) == "faiss":
        raise ValueError("FAISS 후보 축소에는 ONNX 인코더를 사용하세요. macOS에서는 Torch와 FAISS의 OpenMP 런타임이 충돌합니다.")
    artifacts = Path(artifacts)
    meta = json.loads((artifacts / "meta.json").read_text(encoding="utf-8"))
    items = pd.read_parquet(artifacts / "items.parquet")
    lat, lon = item_coords(items)
    if backend.startswith("onnx-"):
        encoder = OnnxEncoder(artifacts / f"encoder_{backend.split('-')[1]}.onnx", threads)
    else:
        from dongne_rec.serving.export_onnx import load_trained

        encoder = TorchEncoder(load_trained(Path(meta["run"])), backend.split("-")[1], threads)
    rec = TwoStageRecommender(np.load(artifacts / "item_emb.npy"), lat, lon, encoder, meta["max_len"], meta["radius_km"], **pipeline_kwargs)
    return rec, items, meta
