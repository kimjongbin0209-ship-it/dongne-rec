"""추천 API (FastAPI). 서빙 프로세스는 torch 없이 ONNX Runtime + FAISS만 써요.

DONGNE_ARTIFACTS=artifacts/NYC/R3/mixed_s0 uvicorn dongne_rec.serving.api:create_app --factory --port 8000

POST /recommend  {"recent_item_ids": [...], "lat": 40.73, "lon": -73.99, "k": 10}
GET  /health
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dongne_rec.data.geo import haversine_km
from dongne_rec.serving.pipeline import load_recommender


class RecommendRequest(BaseModel):
    recent_item_ids: list[int] = Field(min_length=1, description="최근 방문 장소 item_idx, 오래된 순 (최근 max_len개만 사용)")
    lat: float = Field(ge=-90, le=90, description="현재 위치 위도")
    lon: float = Field(ge=-180, le=180, description="현재 위치 경도")
    k: int = Field(10, ge=1, le=100)
    radius_km: float | None = Field(None, gt=0, le=50, description="동네 반경. 없으면 학습 때 반경")


class RecommendedItem(BaseModel):
    item_id: int
    venue_id: str
    category: str
    distance_km: float
    score: float


class RecommendResponse(BaseModel):
    items: list[RecommendedItem]
    debug: dict


def create_app(artifacts: str | Path | None = None, backend: str | None = None, **pipeline_kwargs) -> FastAPI:
    artifacts = Path(artifacts or os.environ.get("DONGNE_ARTIFACTS", "artifacts/NYC/R3/mixed_s0"))
    backend = backend or os.environ.get("DONGNE_BACKEND", "onnx-int8")
    rec, items, meta = load_recommender(artifacts, backend, **pipeline_kwargs)
    items = items.set_index("item_idx")
    app = FastAPI(title="dongne-rec", version="0.1.0", description="동네 반경 제약 순차 추천")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": meta["model"], "backend": backend, "n_items": rec.n_items, "radius_km": rec.radius_km}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> dict:
        recent = np.asarray(req.recent_item_ids, dtype=np.int64)
        if recent.min() < 1 or recent.max() > rec.n_items:
            raise HTTPException(status_code=422, detail=f"recent_item_ids는 1..{rec.n_items} 범위여야 해요")
        r = rec.recommend(recent, req.lat, req.lon, k=req.k, radius_km=req.radius_km)
        picked = items.loc[r.items]
        dist = haversine_km(req.lat, req.lon, picked["lat"].to_numpy(), picked["lon"].to_numpy())
        return {
            "items": [
                {"item_id": int(i), "venue_id": v, "category": c, "distance_km": round(float(d), 3), "score": float(s)}
                for i, v, c, d, s in zip(r.items, picked["venue_id"], picked["category"], dist, r.scores)
            ],
            "debug": {"n_geo": r.n_geo, "n_candidates": r.n_candidates, "radius_used_km": r.radius_km,
                      "latency_ms": {k: round(v, 3) for k, v in r.timings_ms.items()}},
        }

    return app
