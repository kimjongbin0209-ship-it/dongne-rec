"""2단계 추천: 지리 필터 → 임베딩 후보 → SASRec 재랭킹.

  Stage 1-a 지리 필터  : 현재 위치 반경 R 안 장소 (BallTree). k개 미만이면 반경을 2배씩 확장
  Stage 1-b 후보 축소  : 반경 안이 N개를 넘으면, Transformer 없이 최근 장소 임베딩 평균으로 상위 N개
  Stage 2   재랭킹     : SASRec 인코더(시퀀스 → h)로 후보 점수 h · e_i, 상위 k개
현재 장소(최근 시퀀스의 마지막)는 오프라인 평가와 같은 규칙으로 후보에서 빼요.

아이템 수만 개 규모에선 전체 점수 계산도 빨라요. 2단계 구조의 의미는 (1) 동네 제약을 hard filter로 강제하고
(2) 카탈로그가 커져도 Stage 2 비용이 후보 수 N에만 비례하게 만드는 거예요.

주의: macOS에서 PyTorch와 FAISS는 각자 OpenMP 런타임을 들고 와서 한 프로세스에 같이 올리면 충돌해요
(OMP Error #15, segfault). 서빙 경로(ONNX + FAISS)는 torch를 import하지 않고, torch 인코더는
retrieval="numpy"로만 써요 (FAISS 사전 필터와 후보 집합이 같아요).
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
    timings_ms: dict[str, float] = field(default_factory=dict)


class TwoStageRecommender:
    def __init__(
        self, item_emb: np.ndarray, lat: np.ndarray, lon: np.ndarray, encoder, max_len: int, radius_km: float,
        n_candidates: int = 200, query_last_n: int = 3, max_radius_expansions: int = 3, retrieval: str = "faiss",
    ):
        self.item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        self.lat, self.lon = lat, lon
        self.encoder, self.max_len, self.radius_km = encoder, max_len, radius_km
        self.n_candidates, self.query_last_n, self.max_radius_expansions = n_candidates, query_last_n, max_radius_expansions
        self.retrieval = retrieval
        self.tree = build_ball_tree(lat, lon)
        if retrieval == "faiss":
            import faiss  # torch와 같은 프로세스에 올리지 않도록 필요할 때만 (모듈 docstring 참고)

            self.faiss = faiss
            self.index = faiss.IndexFlatIP(self.item_emb.shape[1])
            self.index.add(self.item_emb[1:])  # FAISS id p ↔ item_idx p+1

    @property
    def n_items(self) -> int:
        return len(self.item_emb) - 1

    def geo_filter(self, lat: float, lon: float, exclude: int, k: int, radius_km: float) -> tuple[np.ndarray, float]:
        query = np.radians([[lat, lon]])
        for _ in range(self.max_radius_expansions + 1):
            items = self.tree.query_radius(query, r=radius_km / EARTH_RADIUS_KM)[0] + 1
            items = items[items != exclude]
            if len(items) >= k:
                break
            radius_km *= 2
        return items, radius_km

    def narrow(self, recent: np.ndarray, geo_items: np.ndarray) -> np.ndarray:
        if len(geo_items) <= self.n_candidates:
            return geo_items
        query = self.item_emb[recent[-self.query_last_n :]].mean(axis=0, keepdims=True)
        if self.retrieval == "faiss":
            params = self.faiss.SearchParameters(sel=self.faiss.IDSelectorBatch(geo_items - 1))
            _, ids = self.index.search(query, self.n_candidates, params=params)
            return ids[0][ids[0] >= 0] + 1
        scores = self.item_emb[geo_items] @ query[0]
        return geo_items[np.argpartition(-scores, self.n_candidates)[: self.n_candidates]]

    def rerank(self, recent: np.ndarray, candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        h = self.encoder(pad_left([recent], self.max_len))[0]
        scores = self.item_emb[candidates] @ h
        top = np.argsort(-scores, kind="stable")[:k]
        return candidates[top], scores[top]

    def recommend(self, recent: np.ndarray, lat: float, lon: float, k: int = 10, radius_km: float | None = None) -> Recommendation:
        recent = np.asarray(recent, dtype=np.int64)[-self.max_len :]
        t0 = time.perf_counter_ns()
        geo_items, radius_used = self.geo_filter(lat, lon, int(recent[-1]), k, radius_km or self.radius_km)
        t1 = time.perf_counter_ns()
        candidates = self.narrow(recent, geo_items)
        t2 = time.perf_counter_ns()
        items, scores = self.rerank(recent, candidates, k)
        t3 = time.perf_counter_ns()
        return Recommendation(
            items, scores, len(geo_items), len(candidates), radius_used, candidates,
            {"geo_filter": (t1 - t0) / 1e6, "narrow": (t2 - t1) / 1e6, "rerank": (t3 - t2) / 1e6, "total": (t3 - t0) / 1e6},
        )


def load_recommender(artifacts: Path, backend: str = "onnx-int8", threads: int = 0, **pipeline_kwargs) -> tuple[TwoStageRecommender, pd.DataFrame, dict]:
    """backend: onnx-fp32 | onnx-int8 | torch-cpu | torch-mps. (추천기, items, meta)를 돌려줘요."""
    if backend.startswith("torch-") and pipeline_kwargs.get("retrieval", "faiss") == "faiss":
        raise ValueError("torch 인코더는 FAISS와 한 프로세스에 둘 수 없어요 (OpenMP 충돌). retrieval='numpy'를 쓰세요.")
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
