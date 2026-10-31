import numpy as np
import pandas as pd
import pytest


def make_raw(rows: list[tuple[int, str, float]], coords: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """(user_id, venue_id, 시작 후 경과 시간[h]) 목록 → load_raw()와 같은 스키마의 DataFrame."""
    coords = coords or {}
    t0 = pd.Timestamp("2012-04-03 18:00:00", tz="UTC")
    df = pd.DataFrame(rows, columns=["user_id", "venue_id", "hours"])
    df["category_id"] = "c-" + df["venue_id"]
    df["category"] = "Cat " + df["venue_id"]
    df["lat"] = [coords.get(v, (40.75, -73.99))[0] for v in df["venue_id"]]
    df["lon"] = [coords.get(v, (40.75, -73.99))[1] for v in df["venue_id"]]
    df["ts_utc"] = t0 + pd.to_timedelta(df.pop("hours"), unit="h")
    df["ts_local"] = (df["ts_utc"] - pd.Timedelta(hours=4)).dt.tz_convert(None)
    return df


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """맨해튼 근처 장소 40곳을 30명이 무작위로 방문하는 데이터 (좌표 약간의 노이즈 포함)."""
    rng = np.random.default_rng(0)
    n_venues = 40
    base = {f"v{v:02d}": (40.75 + rng.normal(0, 0.03), -73.99 + rng.normal(0, 0.03)) for v in range(n_venues)}
    rows = []
    for u in range(1, 31):
        hours = np.cumsum(rng.exponential(10, rng.integers(8, 25)))
        rows += [(u, f"v{rng.integers(n_venues):02d}", h) for h in hours]
    df = make_raw(rows, base)
    df["lat"] += rng.normal(0, 1e-5, len(df))
    df["lon"] += rng.normal(0, 1e-5, len(df))
    return df.sample(frac=1, random_state=0)  # 원본처럼 유저가 섞인 순서
