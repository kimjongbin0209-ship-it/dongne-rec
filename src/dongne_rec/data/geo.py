"""거리 계산과 반경 이웃 인덱스.

아이템 ID는 1..N (0 = padding) 이므로, 아이템 단위 배열은 item_idx로 바로
인덱싱할 수 있게 길이 N+1로 맞추고 0번 칸은 비워 둬요.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from dongne_rec.config import city_dir, load_config

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def item_coords(items: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """items(item_idx, lat, lon) → item_idx로 인덱싱되는 lat, lon 배열 (길이 N+1, 0번은 NaN)."""
    n = int(items["item_idx"].max())
    lat = np.full(n + 1, np.nan)
    lon = np.full(n + 1, np.nan)
    lat[items["item_idx"].to_numpy()] = items["lat"].to_numpy()
    lon[items["item_idx"].to_numpy()] = items["lon"].to_numpy()
    return lat, lon


def build_ball_tree(lat: np.ndarray, lon: np.ndarray) -> BallTree:
    """item 1..N 좌표로 haversine BallTree를 만들어요. 트리 내부 위치 p ↔ item_idx p+1."""
    return BallTree(np.radians(np.column_stack([lat[1:], lon[1:]])), metric="haversine")


def count_within(tree: BallTree, lat, lon, radius_km: float) -> np.ndarray:
    """각 질의 좌표에서 반경 안 아이템 수 (거리 0인 자기 자신 포함)."""
    query = np.radians(np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)]))
    return tree.query_radius(query, r=radius_km / EARTH_RADIUS_KM, count_only=True)


def radius_neighbors_csr(
    lat: np.ndarray, lon: np.ndarray, radius_km: float, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """item_idx별 반경 이웃 CSR: neighbors(i) = indices[indptr[i]:indptr[i+1]] (자기 자신 포함, 오름차순).

    lat, lon은 item_coords()가 만든 길이 N+1 배열이에요.
    """
    tree = build_ball_tree(lat, lon)
    n = len(lat) - 1
    r = radius_km / EARTH_RADIUS_KM
    rows: list[np.ndarray] = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        query = np.radians(np.column_stack([lat[1 + start : 1 + stop], lon[1 + start : 1 + stop]]))
        rows.extend(np.sort(nb).astype(np.int32) + 1 for nb in tree.query_radius(query, r=r))
    counts = np.array([len(x) for x in rows], dtype=np.int64)
    indptr = np.concatenate([[0, 0], np.cumsum(counts)]).astype(np.int64)
    indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return indptr, indices


def neighbors_path(cfg: dict, radius_km: float) -> Path:
    return city_dir(cfg, "processed_dir") / f"geo_neighbors_R{radius_km:g}.npz"


def load_neighbors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["indptr"], z["indices"]


def build_neighbors(cfg: dict, radius_km: float) -> Path:
    items = pd.read_parquet(city_dir(cfg, "processed_dir") / "items.parquet")
    lat, lon = item_coords(items)
    indptr, indices = radius_neighbors_csr(lat, lon, radius_km)
    out = neighbors_path(cfg, radius_km)
    np.savez_compressed(out, indptr=indptr, indices=indices, radius_km=radius_km)
    return out


@dataclass
class GeoIndex:
    """장소 좌표 + 반경 R 이웃. 평가 프로토콜(E2/E4)과 지리 기반 모델이 같이 써요."""

    radius_km: float
    lat: np.ndarray
    lon: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray

    @property
    def n_items(self) -> int:
        return len(self.lat) - 1

    def neighbors(self, item: int) -> np.ndarray:
        return self.indices[self.indptr[item] : self.indptr[item + 1]]

    def mask(self, items: np.ndarray) -> np.ndarray:
        """(len(items), N+1) bool: 행 b = items[b] 반경 안에 있는 아이템."""
        out = np.zeros((len(items), self.n_items + 1), dtype=bool)
        for b, i in enumerate(items):
            out[b, self.neighbors(i)] = True
        return out


def load_geo_index(cfg: dict, radius_km: float) -> GeoIndex:
    """이웃 CSR이 없으면 만들어서 저장한 뒤 불러와요."""
    path = neighbors_path(cfg, radius_km)
    if not path.exists():
        build_neighbors(cfg, radius_km)
    lat, lon = item_coords(pd.read_parquet(city_dir(cfg, "processed_dir") / "items.parquet"))
    indptr, indices = load_neighbors(path)
    return GeoIndex(radius_km, lat, lon, indptr, indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="반경 R 이웃 CSR 생성")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    radius_km = args.radius_km if args.radius_km is not None else cfg["geo"]["radius_km"]
    if radius_km is None:
        parser.error("반경이 없어요. --radius-km을 주거나 config의 geo.radius_km을 채워 주세요 (make eda 결과 참고).")

    out = build_neighbors(cfg, radius_km)
    indptr, indices = load_neighbors(out)

    deg = np.diff(indptr)[1:]
    print(
        f"[geo] {cfg['city']} R={radius_km:g}km: items={len(deg):,} "
        f"이웃 수 median={np.median(deg):.0f} p90={np.percentile(deg, 90):.0f} max={deg.max():,} "
        f"| CSR {indices.nbytes / 1e6:.1f}MB → {out}"
    )


if __name__ == "__main__":
    main()
