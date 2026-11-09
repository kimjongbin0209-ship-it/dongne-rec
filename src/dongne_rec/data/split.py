"""이용자별 leave-one-out과 전역 시간 분할.

유저 시퀀스 s_1..s_T에 대해
  train: s_1..s_{T-2}                (모델 학습과 인기도·유사도 통계에 쓰는 유일한 데이터)
  valid: 입력 s_1..s_{T-2} → 정답 s_{T-1}
  test : 입력 s_1..s_{T-1} → 정답 s_T  (valid 정답은 입력으로만 쓰고 학습에는 안 씀)
현재 위치는 입력의 마지막 장소 좌표예요.

전역 시간 분할은 학습 종료 전에 만든 카탈로그를 고정하고, 기간마다
학습 이용자의 첫 추천 가능 시점을 평가합니다. 미등록 정답의 ID는 0입니다.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dongne_rec.config import city_dir
from dongne_rec.storage import arrays_fingerprint, fingerprint


@dataclass
class Queries:
    stage: str
    users: np.ndarray
    histories: list[np.ndarray]  # 정답 직전까지의 입력 시퀀스 (시간순)
    targets: np.ndarray
    gap_h: np.ndarray            # 입력의 마지막 체크인 → 정답까지 시간
    last_items: np.ndarray = field(init=False)  # 현재 위치를 정하는 장소 = histories[i][-1]
    is_repeat: np.ndarray = field(init=False)   # 정답이 입력 시퀀스에 이미 있는지

    def __post_init__(self) -> None:
        self.last_items = np.array([h[-1] for h in self.histories], dtype=np.int64)
        self.is_repeat = np.array([np.isin(t, h) for t, h in zip(self.targets, self.histories)], dtype=bool)

    def __len__(self) -> int:
        return len(self.targets)


@dataclass
class Split:
    n_items: int
    train: list[np.ndarray]
    valid: Queries
    test: Queries
    strategy: str = "leave-one-out"
    cutoffs: tuple[str, str] | None = None

    def queries(self, stage: str) -> Queries:
        return {"valid": self.valid, "test": self.test}[stage]


def leave_one_out(inter: pd.DataFrame, n_items: int) -> Split:
    inter = inter.sort_values(["user_idx", "ts_utc"], kind="stable")
    users, starts = np.unique(inter["user_idx"].to_numpy(), return_index=True)
    seqs = np.split(inter["item_idx"].to_numpy().astype(np.int64), starts[1:])
    hours = (inter["ts_utc"] - inter["ts_utc"].min()) / pd.Timedelta(hours=1)
    times = np.split(hours.to_numpy(), starts[1:])
    if min(len(s) for s in seqs) < 3:
        raise ValueError("leave-one-out에는 유저당 체크인이 최소 3개 필요해요.")

    valid = Queries("valid", users, [s[:-2] for s in seqs], np.array([s[-2] for s in seqs]),
                    np.array([t[-2] - t[-3] for t in times]))
    test = Queries("test", users, [s[:-1] for s in seqs], np.array([s[-1] for s in seqs]),
                   np.array([t[-1] - t[-2] for t in times]))
    return Split(n_items, [s[:-2] for s in seqs], valid, test)


def global_time_boundaries(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    options = cfg["split"]
    train_end = pd.to_datetime(options["train_end"], utc=True)
    valid_end = pd.to_datetime(options["valid_end"], utc=True)
    if pd.isna(train_end) or pd.isna(valid_end) or train_end >= valid_end:
        raise ValueError("전역 시간 분할은 train_end < valid_end여야 합니다.")
    return train_end, valid_end


def global_time_split(
    train: pd.DataFrame, holdout: pd.DataFrame, n_items: int,
    train_end: pd.Timestamp, valid_end: pd.Timestamp, *, collapse: bool = True,
) -> Split:
    if train.empty or holdout.empty or train["ts_utc"].max() >= train_end or holdout["ts_utc"].min() < train_end:
        raise ValueError("학습과 holdout의 시각이 전역 시간 경계와 일치해야 합니다.")
    if not train["item_idx"].between(1, n_items).all() or not holdout["item_idx"].between(0, n_items).all():
        raise ValueError("전역 분할의 장소 ID가 학습 카탈로그와 일치해야 합니다.")

    def sequence(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ordered = group.sort_values("ts_utc", kind="stable")
        return ordered["item_idx"].to_numpy(dtype=np.int64), ordered["ts_utc"].to_numpy(dtype="datetime64[ns]")

    training = {user: sequence(group) for user, group in train.groupby("user_idx", sort=True)}
    folds = {"valid": [], "test": []}
    boundary = valid_end.to_datetime64()
    for user, group in holdout.groupby("user_idx", sort=True):
        if user not in training:
            raise ValueError("holdout은 학습 카탈로그의 이용자로 제한해야 합니다.")
        prefix, prefix_times = training[user]
        items, times = sequence(group)
        middle = times.searchsorted(boundary)
        for stage, indices in (("valid", range(middle)), ("test", range(middle, len(items)))):
            for i in indices:
                # 같은 시각에 기록된 다른 체크인은 이 요청의 이력에 포함하지 않습니다.
                prior = times.searchsorted(times[i], side="left")
                known = items[:prior] > 0
                history = np.concatenate([prefix, items[:prior][known]])
                history_times = np.concatenate([prefix_times, times[:prior][known]])
                if collapse:
                    keep = np.r_[True, history[1:] != history[:-1]]
                    history, history_times = history[keep], history_times[keep]
                target = items[i]
                if target == history[-1]:
                    continue
                gap = (times[i] - history_times[-1]) / np.timedelta64(1, "h")
                folds[stage].append((user, history, target, gap))
                break

    queries = {}
    for stage, rows in folds.items():
        if not rows:
            raise ValueError(f"전역 시간 분할에 {stage} 쿼리가 없습니다.")
        users, histories, targets, gaps = zip(*rows)
        queries[stage] = Queries(stage, np.array(users), list(histories), np.array(targets), np.array(gaps))
    return Split(
        n_items, [items for items, _ in training.values()], queries["valid"], queries["test"], "global-time",
        (train_end.isoformat(), valid_end.isoformat()),
    )


def split_fingerprint(split: Split) -> str:
    parts = {"train": arrays_fingerprint(split.train)}
    for stage in ("valid", "test"):
        q = split.queries(stage)
        parts[stage] = arrays_fingerprint([q.users, q.targets, q.gap_h, *q.histories])
    identity = {"strategy": split.strategy, "n_items": split.n_items, "parts": parts}
    if split.cutoffs is not None:
        identity["cutoffs"] = split.cutoffs
    return fingerprint(identity)


def read_parquet_input(path: Path, *, columns: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    content = path.read_bytes()
    return pd.read_parquet(io.BytesIO(content), columns=columns), hashlib.sha256(content).hexdigest()


def load_split_inputs(cfg: dict) -> tuple[Split, pd.DataFrame, dict[str, str]]:
    proc = city_dir(cfg, "processed_dir")
    inter, inter_hash = read_parquet_input(proc / "interactions.parquet", columns=["user_idx", "item_idx", "ts_utc"])
    items, items_hash = read_parquet_input(proc / "items.parquet")
    files = {"interactions.parquet": inter_hash, "items.parquet": items_hash}
    n_items = int(items["item_idx"].max())
    strategy = cfg.get("split", {}).get("strategy", "leave-one-out")
    if strategy == "leave-one-out":
        return leave_one_out(inter, n_items), items, files
    if strategy != "global-time":
        raise ValueError(f"지원하지 않는 분할입니다. {strategy}")
    holdout, files["holdout.parquet"] = read_parquet_input(proc / "holdout.parquet")
    train_end, valid_end = global_time_boundaries(cfg)
    manifest = json.loads((proc / "stats.json").read_text(encoding="utf-8"))["split"]
    if manifest["train_end"] != train_end.isoformat() or manifest["inputs"] != files:
        raise ValueError("전역 시간 분할의 입력 파일이 일치하지 않습니다. 같은 설정으로 make data를 실행하세요.")
    split = global_time_split(
        inter, holdout, n_items, train_end, valid_end, collapse=cfg["collapse_consecutive_duplicates"],
    )
    return split, items, files


def load_split(cfg: dict) -> Split:
    return load_split_inputs(cfg)[0]
