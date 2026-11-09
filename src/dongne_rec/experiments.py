"""Evaluation identity, run metadata and lookup for completed experiments."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from uuid import uuid4

from dongne_rec.data.geo import GeoIndex, load_geo_index
from dongne_rec.data.split import Split, load_split_inputs, split_fingerprint
from dongne_rec.eval.protocols import candidate_cache_key, load_or_sample_candidates
from dongne_rec.storage import fingerprint


def load_evaluation(cfg: dict, ecfg: dict, radius: float) -> tuple[Split, GeoIndex, dict, dict]:
    split, items, files = load_split_inputs(cfg)
    geo = load_geo_index(cfg, radius, items=items)
    data_id = fingerprint(files)
    split_id = split_fingerprint(split)
    n_neg, seed = ecfg["n_sampled_negatives"], ecfg["seed"]
    key = candidate_cache_key(geo, n_neg, seed, split_id, data_id)
    candidates = load_or_sample_candidates(cfg, split, geo, n_neg, seed, cache_key=key)
    identity = {
        "schema_version": 1,
        "city": cfg["city"],
        "radius_km": float(radius),
        "data_id": data_id,
        "split_id": split_id,
        "candidate_cache_key": key,
        "ks": sorted(set(ecfg["ks"])),
    }
    return split, geo, candidates, {
        **identity, "evaluation_id": fingerprint(identity), "processed_files": files, "split_strategy": split.strategy,
        **({"split_cutoffs": list(split.cutoffs)} if split.cutoffs is not None else {}),
    }


def environment_metadata() -> dict:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        cpu = brand.stdout.strip() or cpu
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    return {
        "python": platform.python_version(),
        "python_build": platform.python_build(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def git_metadata() -> dict:
    repo = Path(__file__).resolve().parents[2]
    git = ["git", "--no-optional-locks", "-C", str(repo)]
    try:
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            [*git, "status", "--porcelain", "--", "src", "configs", "Makefile", "pyproject.toml", "requirements-*.txt"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def attempt_id(prefix: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{prefix}-{now:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:12]}"


def runs_root(results_dir: Path, city: str, radius: float) -> Path:
    return results_dir / "runs" / city / f"R{radius:g}"


def new_run(
    kind: str, cfg: dict, ecfg: dict, evaluation: dict, *,
    scfg: dict | None = None, sampler: str | None = None, seed: int | None = None,
    tag: str | None = None, device: str | None = None,
) -> tuple[Path, dict]:
    city, radius = evaluation["city"], evaluation["radius_km"]
    data_cfg = {**cfg, "geo": {**cfg["geo"], "radius_km": radius}}
    variant_id = fingerprint({
        "kind": kind,
        "evaluation_id": evaluation["evaluation_id"],
        "data": {k: v for k, v in data_cfg.items() if k not in {"raw_dir", "processed_dir", "eda_dir"}},
        "eval": {k: v for k, v in ecfg.items() if k != "results_dir"},
        "sasrec": scfg,
        "sampler": sampler,
        "tag": tag,
    })
    experiment_id = fingerprint({"variant_id": variant_id, "seed": seed})
    now = datetime.now(timezone.utc)
    name = kind if sampler is None else f"{sampler}_s{seed}"
    run_id = attempt_id(f"{name}-{experiment_id[:12]}", now)
    out = runs_root(Path(ecfg["results_dir"]), city, radius) / run_id
    return out, {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "created_at": now.isoformat(),
        "kind": kind,
        "sampler": sampler,
        "seed": seed,
        "tag": tag,
        "device": device,
        "evaluation": evaluation,
        "configs": {"data": data_cfg, "eval": ecfg, "sasrec": scfg},
        "environment": environment_metadata(),
        "git": git_metadata(),
    }


def saved_runs(results_dir: Path, city: str, radius: float) -> list[tuple[Path, dict]]:
    runs = []
    for manifest in runs_root(results_dir, city, radius).glob("*/metadata.json"):
        run = manifest.parent
        if run.name.startswith(".pending-"):
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        evaluation = record["evaluation"]
        if evaluation["city"] == city and evaluation["radius_km"] == radius:
            runs.append((run, record))
    return sorted(runs, key=lambda pair: (pair[1]["created_at"], pair[0].name))
