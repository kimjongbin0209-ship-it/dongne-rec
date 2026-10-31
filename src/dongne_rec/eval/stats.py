"""쿼리 단위 paired bootstrap."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(base: np.ndarray, other: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """같은 쿼리들에 대한 두 모델의 쿼리별 지표 → (other − base) 평균 차이, 95% 신뢰구간, 양측 p.

    쿼리(유저)를 복원추출해 평균 차이의 분포를 만들어요. 시드 간 분산은 포함하지 않으니
    SASRec은 쿼리별로 시드 평균을 낸 값을 넣고, 시드 표준편차는 따로 보고해요.
    """
    diff = np.asarray(other, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    base_mean = float(np.mean(base))
    scale = 1 / base_mean if base_mean else float("nan")
    return {
        "base": base_mean,
        "diff": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(p),
        "rel": float(diff.mean()) * scale,
        "rel_lo": float(lo) * scale,
        "rel_hi": float(hi) * scale,
    }
