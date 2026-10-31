"""시퀀스 유틸 (torch 없이 — 서빙 프로세스가 torch를 import하지 않게)."""
from __future__ import annotations

import numpy as np


def pad_left(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    """시퀀스마다 최근 max_len개를 오른쪽 정렬, 앞은 0(padding)."""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for b, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s):
            out[b, -len(s):] = s
    return out
