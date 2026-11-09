from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    """YAML config를 읽고, None이 아닌 override 값으로 최상위 키를 덮어써요."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def city_dir(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg[key]) / cfg["city"]


def resolve_radius(cfg: dict[str, Any], override: float | None = None) -> float:
    value = cfg["geo"]["radius_km"] if override is None else override
    radius = float(value) if value is not None else float("nan")
    if not isfinite(radius) or radius < 0:
        raise ValueError("반경은 0 이상의 유한한 수여야 합니다. --radius-km 또는 geo.radius_km을 확인하세요.")
    return radius
