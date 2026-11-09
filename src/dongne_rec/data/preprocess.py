"""Foursquare TSMC2014 체크인 전처리.

원본 CSV → data/processed/{city}/
  interactions.parquet  user_idx, item_idx, ts_utc, ts_local  (유저별 시간순)
  items.parquet         item_idx, venue_id, lat, lon, category_id, category, n_checkins
  users.parquet         user_idx, user_id
  stats.json            단계별 행/유저/아이템 수 (README 표 1 근거)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from dongne_rec.config import city_dir, load_config
from dongne_rec.data.geo import haversine_km
from dongne_rec.data.split import global_time_boundaries

COLUMN_MAP = {
    "userId": "user_id",
    "venueId": "venue_id",
    "venueCategoryId": "category_id",
    "venueCategory": "category",
    "latitude": "lat",
    "longitude": "lon",
    "timezoneOffset": "tz_offset_min",
    "utcTimestamp": "utc_timestamp",
}
RAW_DTYPES = {
    "userId": "int64",
    "venueId": "str",
    "venueCategoryId": "str",
    "venueCategory": "str",
    "latitude": "float64",
    "longitude": "float64",
    "timezoneOffset": "int64",
    "utcTimestamp": "str",
}
TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"  # 예: "Tue Apr 03 18:00:09 +0000 2012"


def raw_path(cfg: dict) -> Path:
    return Path(cfg["raw_dir"]) / f"dataset_TSMC2014_{cfg['city']}.csv"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Kaggle CSV(UTF-8, 헤더 있음)를 읽어 ts_utc(tz-aware)와 ts_local(현지 시각, naive)을 붙여요."""
    df = pd.read_csv(path, encoding="utf-8", dtype=RAW_DTYPES).rename(columns=COLUMN_MAP)
    df["ts_utc"] = pd.to_datetime(df.pop("utc_timestamp"), format=TIMESTAMP_FORMAT, utc=True)
    df["ts_local"] = (df["ts_utc"] + pd.to_timedelta(df.pop("tz_offset_min"), unit="min")).dt.tz_convert(None)
    return df


def collapse_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """같은 유저의 같은 장소 연속 체크인은 첫 번째만 남겨요. df는 (user_id, ts_utc) 순으로 정렬돼 있어야 해요."""
    repeated = df["user_id"].eq(df["user_id"].shift()) & df["venue_id"].eq(df["venue_id"].shift())
    return df[~repeated]


def kcore(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """모든 유저 ≥ min_user, 모든 장소 ≥ min_item 체크인이 될 때까지 반복해서 걸러요."""
    while True:
        keep = (df.groupby("user_id")["venue_id"].transform("size") >= min_user) & (
            df.groupby("venue_id")["user_id"].transform("size") >= min_item
        )
        if keep.all():
            return df
        df = df[keep]


def filter_until_stable(df: pd.DataFrame, min_user: int, min_item: int, collapse: bool) -> tuple[pd.DataFrame, int]:
    """k-core로 중간 체크인이 빠지면 연속 중복이 새로 생길 수 있어서(A→B→A에서 B 제거 → A→A)
    연속 중복 제거와 k-core를 변화가 없을 때까지 번갈아 적용해요. (결과, 반복 횟수)를 돌려줘요."""
    rounds = 0
    while True:
        rounds += 1
        n = len(df)
        if collapse:
            df = collapse_consecutive_duplicates(df)
        df = kcore(df, min_user, min_item)
        if len(df) == n:
            return df, rounds


def build_items(raw: pd.DataFrame, kept: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """남은 장소의 대표 좌표(원본 체크인 좌표의 중앙값)와 최빈 카테고리로 items 테이블을 만들어요.
    item_idx는 venue_id 정렬 순서로 1..N. 두 번째 반환값은 체크인 좌표와 대표 좌표 사이 거리(km)."""
    rows = raw[raw["venue_id"].isin(kept["venue_id"].unique())]
    coords = rows.groupby("venue_id")[["lat", "lon"]].median()
    category = (
        rows.groupby(["venue_id", "category_id", "category"]).size().rename("n").reset_index()
        .sort_values(["venue_id", "n", "category_id"], ascending=[True, False, True])
        .drop_duplicates("venue_id")
        .set_index("venue_id")[["category_id", "category"]]
    )
    n_checkins = kept["venue_id"].value_counts().rename("n_checkins")
    items = coords.join(category).join(n_checkins).sort_index().reset_index()
    items.insert(0, "item_idx", np.arange(1, len(items) + 1, dtype=np.int32))

    median = coords.loc[rows["venue_id"]]
    jitter_km = haversine_km(rows["lat"], rows["lon"], median["lat"], median["lon"])
    return items, jitter_km


def _snapshot(step: str, df: pd.DataFrame) -> dict:
    return {"step": step, "rows": len(df), "users": int(df["user_id"].nunique()), "items": int(df["venue_id"].nunique())}


def preprocess(
    raw: pd.DataFrame, min_user: int, min_item: int, collapse: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = raw.sort_values(["user_id", "ts_utc"], kind="stable", ignore_index=True)  # 같은 시각은 원본 순서 유지
    steps = [_snapshot("raw", df)]
    if collapse:
        df = collapse_consecutive_duplicates(df)
        steps.append(_snapshot("collapse_consecutive_duplicates", df))
    df, rounds = filter_until_stable(df, min_user, min_item, collapse)
    steps.append(_snapshot(f"kcore(user>={min_user}, item>={min_item})", df))

    items, jitter_km = build_items(raw, df)
    users = pd.DataFrame({"user_id": np.sort(df["user_id"].unique())})
    users.insert(0, "user_idx", np.arange(len(users), dtype=np.int32))

    df = df.reset_index(drop=True)
    interactions = pd.DataFrame({
        "user_idx": df["user_id"].map(users.set_index("user_id")["user_idx"]).astype(np.int32),
        "item_idx": df["venue_id"].map(items.set_index("venue_id")["item_idx"]).astype(np.int32),
        "ts_utc": df["ts_utc"],
        "ts_local": df["ts_local"],
    })

    n_users, n_items, n_inter = len(users), len(items), len(interactions)
    stats = {
        "steps": steps,
        "filter_rounds": rounds,
        "final": {
            "users": n_users,
            "items": n_items,
            "interactions": n_inter,
            "density": n_inter / (n_users * n_items),
            "categories": int(items["category"].nunique()),
            "period_utc": [str(interactions["ts_utc"].min()), str(interactions["ts_utc"].max())],
        },
        "coord_jitter_km": {
            "p50": float(np.quantile(jitter_km, 0.5)),
            "p99": float(np.quantile(jitter_km, 0.99)),
            "max": float(jitter_km.max()),
        },
        "params": {"min_user": min_user, "min_item": min_item, "collapse_consecutive_duplicates": collapse},
    }
    return interactions, items, users, stats


def run(cfg: dict) -> dict:
    raw = load_raw(raw_path(cfg))
    strategy = cfg.get("split", {}).get("strategy", "leave-one-out")
    if strategy not in ("leave-one-out", "global-time"):
        raise ValueError(f"지원하지 않는 분할입니다. {strategy}")
    cutoffs = global_time_boundaries(cfg) if strategy == "global-time" else None
    train = raw if cutoffs is None else raw[raw["ts_utc"] < cutoffs[0]]
    if train.empty:
        raise ValueError("학습 기간에 체크인이 없습니다.")
    collapse = cfg["collapse_consecutive_duplicates"]
    interactions, items, users, stats = preprocess(
        train, cfg["min_user_checkins"], cfg["min_item_checkins"], collapse,
    )
    stats["source"] = raw_path(cfg).name
    tables = {"interactions.parquet": interactions, "items.parquet": items, "users.parquet": users}
    if cutoffs is not None:
        train_end, valid_end = cutoffs
        ordered = raw.sort_values(["user_id", "ts_utc"], kind="stable")
        if collapse:
            ordered = collapse_consecutive_duplicates(ordered)
        later = ordered[ordered["ts_utc"] >= train_end]
        eligible = later[later["user_id"].isin(users["user_id"])]
        holdout = pd.DataFrame({
            "user_idx": eligible["user_id"].map(users.set_index("user_id")["user_idx"]).astype(np.int32),
            "item_idx": eligible["venue_id"].map(items.set_index("venue_id")["item_idx"]).fillna(0).astype(np.int32),
            "ts_utc": eligible["ts_utc"],
        })
        tables["holdout.parquet"] = holdout
        stats["split"] = {
            "strategy": strategy, "train_end": train_end.isoformat(), "valid_end": valid_end.isoformat(),
            "holdout_rows": len(holdout), "holdout_users": int(holdout["user_idx"].nunique()),
            "excluded_holdout_users": int(later["user_id"].nunique() - holdout["user_idx"].nunique()),
            "unknown_item_rows": int(holdout["item_idx"].eq(0).sum()),
        }

    out = city_dir(cfg, "processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out, prefix=".pending-") as temp:
        staging = Path(temp)
        for name, frame in tables.items():
            frame.to_parquet(staging / name, index=False)
        if cutoffs is not None:
            stats["split"]["inputs"] = {
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in ("interactions.parquet", "items.parquet", "holdout.parquet")
            }
        (staging / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in (*tables, "stats.json"):
            (staging / name).replace(out / name)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Foursquare 체크인 전처리")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--city")
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    stats = run(cfg)
    for s in stats["steps"]:
        print(f"[preprocess] {s['step']:<36} rows={s['rows']:>8,} users={s['users']:>6,} items={s['items']:>7,}")
    f = stats["final"]
    print(
        f"[preprocess] {cfg['city']} 완료: users={f['users']:,} items={f['items']:,} "
        f"interactions={f['interactions']:,} density={f['density']:.4%} (filter rounds={stats['filter_rounds']})"
        f" → {city_dir(cfg, 'processed_dir')}"
    )


if __name__ == "__main__":
    main()
