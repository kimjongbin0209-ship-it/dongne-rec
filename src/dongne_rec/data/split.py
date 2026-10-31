"""시간순 leave-one-out 분할.

유저 시퀀스 s_1..s_T에 대해
  train: s_1..s_{T-2}                (모델 학습과 인기도·유사도 통계에 쓰는 유일한 데이터)
  valid: 입력 s_1..s_{T-2} → 정답 s_{T-1}
  test : 입력 s_1..s_{T-1} → 정답 s_T  (valid 정답은 입력으로만 쓰고 학습에는 안 씀)
현재 위치는 입력의 마지막 장소 좌표예요.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dongne_rec.config import city_dir


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


def load_split(cfg: dict) -> Split:
    proc = city_dir(cfg, "processed_dir")
    inter = pd.read_parquet(proc / "interactions.parquet", columns=["user_idx", "item_idx", "ts_utc"])
    n_items = int(pd.read_parquet(proc / "items.parquet", columns=["item_idx"])["item_idx"].max())
    return leave_one_out(inter, n_items)
