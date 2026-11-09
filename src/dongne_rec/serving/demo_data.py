"""Export sanitized /recommend examples for the static HTTPS demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from dongne_rec.data.geo import EARTH_RADIUS_KM, haversine_km
from dongne_rec.serving.api import create_app
from dongne_rec.storage import write_json

RADII = (2, 3, 5)
SCENARIOS = (
    ("midtown", "미드타운", "MANHATTAN", 40.7549, -73.9840,
     (("Coffee Shop", "카페"), ("Bookstore", "서점"), ("Park", "공원"))),
    ("east-village", "이스트 빌리지", "MANHATTAN", 40.7270, -73.9855,
     (("Bakery", "베이커리"), ("Record Shop", "음반점"), ("Pizza Place", "피자 가게"))),
    ("williamsburg", "윌리엄스버그", "BROOKLYN", 40.7140, -73.9580,
     (("Coffee Shop", "카페"), ("Thrift / Vintage Store", "빈티지 숍"), ("Bar", "바"))),
)


def export_examples(artifact: Path) -> dict:
    catalog = pd.read_parquet(artifact / "items.parquet").set_index("item_idx")
    policies = ("full", "faiss")
    clients = {policy: TestClient(create_app(artifact, "onnx-int8", retrieval=policy, n_candidates=200,
                                            radius_mode="fixed")) for policy in policies}
    health = clients["full"].get("/health").raise_for_status().json()
    profiles = []
    for key, label, borough, lat, lon, stops in SCENARIOS:
        # Choose nearby catalog examples by category, independent of every user's visit log.
        history = []
        for category, _ in stops:
            matching = catalog[catalog["category"] == category]
            history.append(int(matching.index[haversine_km(lat, lon, matching["lat"], matching["lon"]).argmin()]))
        profile = {"key": key, "label": label, "borough": borough, "history": [name for _, name in stops], "runs": {}}
        aliases = {}
        for radius in RADII:
            results = {}
            for policy, client in clients.items():
                response = client.post("/recommend", json={"recent_item_ids": history, "lat": lat, "lon": lon,
                                                            "k": 10, "radius_km": radius}).raise_for_status().json()
                debug = response["debug"]
                shown = []
                for item in response["items"]:
                    item_id = item["item_id"]
                    venue = catalog.loc[item_id]
                    category = item["category"]
                    private = any(word in category.lower() for word in ("private", "residential"))
                    xy = None
                    if not private:
                        # Publish coarse offsets on a schematic, without venue IDs or absolute coordinates.
                        xy = [round(EARTH_RADIUS_KM * math.radians(venue["lon"] - lon) * math.cos(math.radians(lat)), 1),
                              round(EARTH_RADIUS_KM * math.radians(venue["lat"] - lat), 1)]
                    shown.append({"ref": aliases.setdefault(item_id, len(aliases) + 1),
                                  "category": "주거 시설" if private else category,
                                  "km": round(item["distance_km"], 1), "xy": xy})
                results[policy] = {"nearby": debug["n_geo"], "candidates": debug["n_candidates"], "items": shown}
            profile["runs"][str(radius)] = results
        profiles.append(profile)
    for client in clients.values():
        client.close()
    return {"model": health["model"], "backend": health["backend"], "radii": RADII, "profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser(description="합성 시나리오의 정적 웹 데모 데이터 생성")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/NYC/R3/mixed_s0"))
    parser.add_argument("--benchmark", type=Path, required=True, help="HTTP 벤치의 http.json")
    parser.add_argument("--output", type=Path, default=Path("web/demo/dist/demo-data.json"))
    args = parser.parse_args()
    examples = export_examples(args.artifacts)
    bench = json.loads(args.benchmark.read_text(encoding="utf-8"))
    examples["benchmark"] = {
        "cpu": bench["environment"]["cpu"], "requests": sum(r["requests"] for r in bench["conditions"]),
        "conditions": [{"policy": r["policy"], "concurrency": r["concurrency"],
                        "p99": round(r["http_ms"]["p99"], 2), "rps": round(r["success_rps"], 1)}
                       for r in bench["conditions"]],
    }
    write_json(args.output, examples)


if __name__ == "__main__":
    main()
