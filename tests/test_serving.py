import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dongne_rec.data.geo import GeoIndex, haversine_km, radius_neighbors_csr
from dongne_rec.data.split import Queries
from dongne_rec.eval.protocols import evaluate
from dongne_rec.serving.bench import run_requests
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
                               n_candidates=n_candidates, **kw)


def test_candidates_stay_in_radius_exclude_current_place_and_are_sorted():
    rec = make_rec(radius_km=0.5, n_candidates=2)  # 기본 전체 재랭킹은 축소 상한과 무관
    lat, lon = coords()
    r = rec.recommend(np.array([10, 30]), lat[30], lon[30], k=5)
    assert set(r.candidates) == {26, 27, 28, 29, 31, 32, 33, 34}
    assert 30 not in r.items and len(r.items) == 5
    assert r.radius_km == 0.5 and r.n_candidates == r.n_geo == 8
    assert (np.diff(r.scores) <= 0).all()
    assert (haversine_km(lat[30], lon[30], lat[r.items], lon[r.items]) <= 0.5).all()


def test_narrowing_keeps_top_n_by_recent_embedding_similarity():
    rec = make_rec(radius_km=3.0, n_candidates=5, query_last_n=2, retrieval="numpy")
    lat, lon = coords()
    recent = np.array([3, 40, 30])
    r = rec.recommend(recent, lat[30], lon[30], k=3)
    geo = r.n_geo
    query = rec.item_emb[[40, 30]].mean(0)
    pool = np.array([i for i in range(1, N + 1) if i != 30 and haversine_km(lat[30], lon[30], lat[i], lon[i]) <= 3.0])
    expected = set(pool[np.argsort(-(rec.item_emb[pool] @ query))[:5]])
    assert geo == len(pool) and r.n_candidates == 5 and set(r.candidates) == expected


def test_radius_expands_when_too_few_places():
    rec = make_rec(radius_km=0.05, radius_mode="expand")
    lat, lon = coords()
    r = rec.recommend(np.array([30]), lat[30], lon[30], k=4)
    assert r.radius_km == pytest.approx(0.4) and len(r.items) == 4 and 30 not in r.items


def test_expansion_returns_last_searched_radius_when_candidates_are_insufficient():
    rec = make_rec(radius_km=0.05, radius_mode="expand", max_radius_expansions=2)
    lat, lon = coords()
    r = rec.recommend(np.array([30]), lat[30], lon[30], k=5)
    assert r.radius_km == pytest.approx(0.2)
    assert set(r.items) == {29, 31}
    assert r.n_geo == r.n_candidates == 2


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
        rec = make_rec(radius_km=3.0, n_candidates=7, retrieval=retrieval)
        results.append([set(rec.recommend(np.array([i, i + 1]), lat[i + 1], lon[i + 1]).candidates) for i in range(1, 50, 7)])
    assert "torch" not in sys.modules
    return results[0] == results[1]


def test_fixed_radius_benchmark_matches_e4_with_ties_and_out_of_radius_targets():
    lat, lon = coords()
    emb = np.ones((N + 1, H), dtype=np.float32)
    radius = 0.15
    rec = TwoStageRecommender(emb, lat, lon, LastItemEncoder(emb), max_len=8, radius_km=radius)
    geo = GeoIndex(radius, lat, lon, *radius_neighbors_csr(lat, lon, radius))
    q = Queries("test", np.arange(2), [np.array([30]), np.array([30])], np.array([29, 50]), np.ones(2))

    class EqualScores:
        name = "ties"

        def score(self, histories, last_items):
            return np.full((len(histories), N + 1), H, dtype=np.float32)

    negatives = np.full((len(q), 1), 55)
    offline = evaluate(EqualScores(), q, geo, {"E1": negatives, "E2": negatives}, ks=(1,))
    e4 = offline[offline["protocol"] == "E4"]
    measured = run_requests(rec, q, warmup=0, k=1)
    for metric in ("ndcg@1", "hr@1", "in_candidates"):
        np.testing.assert_array_equal(measured[metric], e4[metric])
    assert measured["ndcg@1"].tolist() == [0.5, 0.0]


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

    client = TestClient(create_app(artifacts, "onnx-int8"))
    policy = client.get("/health").json()["pipeline"]
    assert (policy["retrieval"], policy["radius_mode"]) == ("full", "fixed")

    lat, lon = coords()
    res = client.post("/recommend", json={"recent_item_ids": [10, 30], "lat": lat[30], "lon": lon[30], "k": 5})
    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) == 5 and all(it["distance_km"] <= 0.5 for it in body["items"])
    assert all(it["item_id"] != 30 for it in body["items"])
    scores = [it["score"] for it in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert body["debug"]["n_geo"] == 8
    assert body["debug"]["n_candidates"] == 8 and not body["debug"]["expanded"]

    empty = client.post("/recommend", json={
        "recent_item_ids": [30], "lat": lat[30], "lon": lon[30], "k": 5, "radius_km": 0.05,
    })
    assert empty.status_code == 200
    assert empty.json()["items"] == [] and empty.json()["debug"]["radius_used_km"] == 0.05

    assert client.post("/recommend", json={"recent_item_ids": [999], "lat": 40.0, "lon": -74.0}).status_code == 422
    assert client.post("/recommend", json={"recent_item_ids": [], "lat": 40.0, "lon": -74.0}).status_code == 422


def test_api_uses_serving_yaml_for_artifact_selection_and_expansion(artifacts, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dongne_rec.serving.api import create_app

    destination = tmp_path / "artifacts" / "TEST" / "R0.5" / "geo_s1"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(artifacts, target_is_directory=True)
    data = tmp_path / "data.yaml"
    data.write_text("city: TEST\ngeo: {radius_km: 0.5}\n", encoding="utf-8")
    serving = tmp_path / "serving.yaml"
    serving.write_text(
        f"artifacts_dir: {tmp_path / 'artifacts'}\nmodel: {{sampler: geo, seed: 1}}\n"
        "pipeline: {retrieval: numpy, n_candidates: 2, radius_mode: expand, max_radius_expansions: 2}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DONGNE_SERVING_CONFIG", str(serving))
    monkeypatch.setenv("DONGNE_DATA_CONFIG", str(data))
    for name in ("DONGNE_ARTIFACTS", "DONGNE_CITY", "DONGNE_RADIUS_KM", "DONGNE_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    client = TestClient(create_app())
    policy = client.get("/health").json()["pipeline"]
    assert policy["retrieval"] == "numpy" and policy["radius_mode"] == "expand"

    lat, lon = coords()
    expanded = client.post("/recommend", json={
        "recent_item_ids": [30], "lat": lat[30], "lon": lon[30], "k": 5, "radius_km": 0.05,
    })
    assert expanded.status_code == 200
    body = expanded.json()
    assert body["debug"]["expanded"] and body["debug"]["radius_used_km"] == pytest.approx(0.2)
    assert {item["item_id"] for item in body["items"]} == {29, 31}
    dense = client.post("/recommend", json={"recent_item_ids": [30], "lat": lat[30], "lon": lon[30], "k": 3}).json()
    assert dense["debug"]["n_geo"] == 8 and len(dense["items"]) == dense["debug"]["n_candidates"] == 2
