from __future__ import annotations

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
