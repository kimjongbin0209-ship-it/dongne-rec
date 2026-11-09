"""FastAPI 추천 API. 기본 파이프라인은 ONNX로 동네 후보 전체를 재랭킹합니다.

make serve SERVING_CONFIG=configs/serving.yaml

POST /recommend  {"recent_item_ids": [...], "lat": 40.73, "lon": -73.99, "k": 10}
GET  /health
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dongne_rec.config import load_config, resolve_radius
from dongne_rec.data.geo import haversine_km
from dongne_rec.serving.pipeline import DEFAULT_SERVING_CONFIG, artifacts_dir, load_recommender, pipeline_settings


class RecommendRequest(BaseModel):
    recent_item_ids: list[int] = Field(min_length=1, description="최근 방문 장소 item_idx, 오래된 순 (최근 max_len개만 사용)")
    lat: float = Field(ge=-90, le=90, description="현재 위치 위도")
    lon: float = Field(ge=-180, le=180, description="현재 위치 경도")
    k: int = Field(10, ge=1, le=100)
    radius_km: float | None = Field(None, gt=0, le=50, description="시작 반경. 생략하면 모델의 기본 반경을 사용합니다.")


class RecommendedItem(BaseModel):
    item_id: int
    venue_id: str
    category: str
    distance_km: float
    score: float


class RecommendResponse(BaseModel):
    items: list[RecommendedItem]
    debug: dict


def local_config(path: str | Path | None, default: str) -> dict:
    if path:
        return load_config(path)
    return load_config(default) if Path(default).is_file() else {}


def configured_artifacts(scfg: dict) -> Path:
    data = local_config(os.environ.get("DONGNE_DATA_CONFIG"), "configs/data.yaml")
    city = os.environ.get("DONGNE_CITY") or data.get("city", "NYC")
    override = os.environ.get("DONGNE_RADIUS_KM")
    radius = resolve_radius(data or {"geo": {"radius_km": 3}}, float(override) if override else None)
    model = scfg.get("model", {})
    return artifacts_dir({"artifacts_dir": scfg.get("artifacts_dir", "artifacts")}, city, radius,
                         model.get("sampler", "mixed"), model.get("seed", 0))


def create_app(
    artifacts: str | Path | None = None, backend: str | None = None,
    *, serving_config: str | Path | None = None, **pipeline_kwargs,
) -> FastAPI:
    scfg = local_config(serving_config or os.environ.get("DONGNE_SERVING_CONFIG"), DEFAULT_SERVING_CONFIG)
    pcfg = {**pipeline_settings(scfg), **pipeline_kwargs}
    artifact_override = artifacts or os.environ.get("DONGNE_ARTIFACTS")
    artifact_path = Path(artifact_override) if artifact_override else configured_artifacts(scfg)
    backend = backend or os.environ.get("DONGNE_BACKEND") or "onnx-int8"
    rec, items, meta = load_recommender(artifact_path, backend, **pcfg)
    items = items.set_index("item_idx")
    app = FastAPI(title="dongne-rec", version="0.1.0", description="동네 반경 제약 순차 추천")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok", "model": meta["model"], "backend": backend, "n_items": rec.n_items,
            "radius_km": rec.radius_km, "pipeline": pcfg,
        }

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> dict:
        recent = np.asarray(req.recent_item_ids, dtype=np.int64)
        if recent.min() < 1 or recent.max() > rec.n_items:
            raise HTTPException(status_code=422, detail=f"recent_item_ids는 1..{rec.n_items} 범위여야 해요")
        requested_radius = rec.radius_km if req.radius_km is None else req.radius_km
        r = rec.recommend(recent, req.lat, req.lon, k=req.k, radius_km=requested_radius)
        picked = items.loc[r.items]
        dist = haversine_km(req.lat, req.lon, picked["lat"].to_numpy(), picked["lon"].to_numpy())
        return {
            "items": [
                {"item_id": int(i), "venue_id": v, "category": c, "distance_km": round(float(d), 3), "score": float(s)}
                for i, v, c, d, s in zip(r.items, picked["venue_id"], picked["category"], dist, r.scores)
            ],
            "debug": {
                "n_geo": r.n_geo, "n_candidates": r.n_candidates,
                "radius_mode": rec.radius_mode, "radius_requested_km": requested_radius,
                "radius_used_km": r.radius_km, "expanded": r.radius_km > requested_radius,
                "latency_ms": {k: round(v, 3) for k, v in r.timings_ms.items()},
            },
        }

    return app
