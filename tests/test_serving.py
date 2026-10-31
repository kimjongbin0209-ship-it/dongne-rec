import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dongne_rec.data.geo import haversine_km
from dongne_rec.serving.pipeline import TwoStageRecommender

N, H = 60, 16


def line_items() -> pd.DataFrame:
    """item i가 위도 40 + i·0.001 (약 111m 간격)에 일렬로."""
    idx = np.arange(1, N + 1)
    return pd.DataFrame({"item_idx": idx, "venue_id": [f"v{i}" for i in idx], "lat": 40 + idx * 0.001,
                         "lon": np.full(N, -74.0), "category": ["Cafe"] * N})


def coords() -> tuple[np.ndarray, np.ndarray]:
    items = line_items()
    return np.concatenate([[np.nan], items["lat"]]), np.concatenate([[np.nan], items["lon"]])


class LastItemEncoder:
    """h = 마지막 장소 임베딩 (파이프라인 로직만 검사하려는 가짜 인코더)."""

    name = "last-item"

    def __init__(self, emb: np.ndarray):
        self.emb = emb

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        return self.emb[seq[:, -1]]


def make_rec(radius_km: float = 0.5, n_candidates: int = 200, **kw) -> TwoStageRecommender:
    emb = np.random.default_rng(0).standard_normal((N + 1, H)).astype(np.float32)
    emb[0] = 0
    lat, lon = coords()
    return TwoStageRecommender(emb, lat, lon, LastItemEncoder(emb), max_len=8, radius_km=radius_km,
                               n_candidates=n_candidates, retrieval="numpy", **kw)


def test_candidates_stay_in_radius_exclude_current_place_and_are_sorted():
    rec = make_rec(radius_km=0.5)  # 반경 안 = ±4개
    lat, lon = coords()
    r = rec.recommend(np.array([10, 30]), lat[30], lon[30], k=5)
    assert set(r.candidates) == {26, 27, 28, 29, 31, 32, 33, 34}
    assert 30 not in r.items and len(r.items) == 5
    assert (np.diff(r.scores) <= 0).all()
    assert (haversine_km(lat[30], lon[30], lat[r.items], lon[r.items]) <= 0.5).all()


def test_narrowing_keeps_top_n_by_recent_embedding_similarity():
    rec = make_rec(radius_km=3.0, n_candidates=5, query_last_n=2)  # 반경 안 거의 전부 → 5개로 축소
    lat, lon = coords()
    recent = np.array([3, 40, 30])
    r = rec.recommend(recent, lat[30], lon[30], k=3)
    geo = r.n_geo
    query = rec.item_emb[[40, 30]].mean(0)
    pool = np.array([i for i in range(1, N + 1) if i != 30 and haversine_km(lat[30], lon[30], lat[i], lon[i]) <= 3.0])
    expected = set(pool[np.argsort(-(rec.item_emb[pool] @ query))[:5]])
    assert geo == len(pool) and r.n_candidates == 5 and set(r.candidates) == expected


def test_radius_expands_when_too_few_places():
    rec = make_rec(radius_km=0.05)  # 반경 안 = 자기 자신뿐 → 2배씩 넓혀 k개 확보
    lat, lon = coords()
    r = rec.recommend(np.array([30]), lat[30], lon[30], k=4)
    assert r.radius_km > 0.05 and len(r.items) == 4 and 30 not in r.items


def test_faiss_selector_matches_numpy_retrieval_in_torch_free_process():
    """FAISS는 torch와 한 프로세스에 두면 OpenMP가 충돌해서(macOS) torch 없는 별도 프로세스에서 검사해요."""
    code = "from test_serving import compare_retrievals; print(compare_retrievals())"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parents[1] / "src")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"


def compare_retrievals() -> bool:
    lat, lon = coords()
    results = []
    for retrieval in ("faiss", "numpy"):
        rec = make_rec(radius_km=3.0, n_candidates=7)
        rec.retrieval = retrieval
        if retrieval == "faiss":
            import faiss

            rec.faiss, rec.index = faiss, faiss.IndexFlatIP(H)
            rec.index.add(rec.item_emb[1:])
        results.append([set(rec.recommend(np.array([i, i + 1]), lat[i + 1], lon[i + 1]).candidates) for i in range(1, 50, 7)])
    assert "torch" not in sys.modules
    return results[0] == results[1]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> Path:
    import torch

    from dongne_rec.models.sasrec import SASRec
    from dongne_rec.serving.export_onnx import export_artifacts

    root = tmp_path_factory.mktemp("serving")
    run = root / "run"
    run.mkdir()
    torch.manual_seed(0)
    cfg = {"max_len": 8, "hidden": H, "n_blocks": 1, "n_heads": 1, "dropout": 0.0}
    torch.save(SASRec(N, **cfg).state_dict(), run / "model.pt")
    (run / "config.json").write_text(json.dumps({"model": cfg}))
    export_artifacts(run, line_items(), root / "art", {"city": "TEST", "radius_km": 0.5, "model": "SASRec-test", "seed": 0})
    return root / "art"


def test_export_is_faithful_and_int8_is_smaller(artifacts):
    meta = json.loads((artifacts / "meta.json").read_text())
    assert meta["max_abs_diff"]["fp32"] < 1e-4
    assert meta["size_mb"]["encoder_int8.onnx"] < meta["size_mb"]["encoder_fp32.onnx"]
    assert np.load(artifacts / "item_emb.npy").shape == (N + 1, H)


def test_api_recommends_nearby_places(artifacts):
    from fastapi.testclient import TestClient

    from dongne_rec.serving.api import create_app

    client = TestClient(create_app(artifacts, "onnx-int8", retrieval="numpy"))
    assert client.get("/health").json()["status"] == "ok"

    lat, lon = coords()
    res = client.post("/recommend", json={"recent_item_ids": [10, 30], "lat": lat[30], "lon": lon[30], "k": 5})
    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) == 5 and all(it["distance_km"] <= 0.5 for it in body["items"])
    assert all(it["item_id"] != 30 for it in body["items"])
    scores = [it["score"] for it in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert body["debug"]["n_geo"] == 8

    assert client.post("/recommend", json={"recent_item_ids": [999], "lat": 40.0, "lon": -74.0}).status_code == 422
    assert client.post("/recommend", json={"recent_item_ids": [], "lat": 40.0, "lon": -74.0}).status_code == 422
