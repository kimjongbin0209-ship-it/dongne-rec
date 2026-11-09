"""Run the stage 3 SASRec training matrix on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import pandas as pd
import torch

from dongne_rec.config import load_config
from dongne_rec.data.split import load_split_inputs, split_fingerprint
from dongne_rec.eval import report
from dongne_rec.eval.protocols import evaluate
from dongne_rec.experiments import load_evaluation, new_run, saved_runs
from dongne_rec.models.sasrec import SASRecScorer
from dongne_rec.serving.export_onnx import load_trained
from dongne_rec.storage import fingerprint, publish_directory, write_json
from dongne_rec.train.trainer import apply_overrides, model_name, run_dir, train_run

CITIES = ("NYC", "TKY")
SEEDS = (0, 1, 2)
SAMPLERS = (("random", 0.0), ("geo", 1.0), ("mixed", 0.5))
LOO_TAG = "phase3-loo"
GLOBAL_TAG = "phase3-global"


@dataclass(frozen=True)
class Job:
    city: str
    radius: int
    sampler: str
    alpha: float
    seed: int


def training_jobs(strategy: str) -> list[Job]:
    if strategy == "global-time":
        conditions = [(3, sampler, alpha) for sampler, alpha in SAMPLERS]
    elif strategy == "leave-one-out":
        conditions = [(radius, sampler, alpha) for radius in (2, 5) for sampler, alpha in SAMPLERS]
        conditions.extend((3, "mixed", alpha) for alpha in (0.25, 0.75))
    else:
        raise ValueError(f"지원하지 않는 분할입니다. {strategy}")
    return [Job(city, radius, sampler, alpha, seed)
            for seed in SEEDS for city in CITIES for radius, sampler, alpha in conditions]


def refresh_r3(data: dict, ecfg: dict, cities: tuple[str, ...]) -> None:
    """Reevaluate the archived R=3 checkpoints under the current cache and provenance."""
    device = torch.device("mps")
    for city in cities:
        cfg = {**data, "city": city}
        split, geo, cands, evaluation = load_evaluation(cfg, ecfg, 3)
        prior_runs = saved_runs(Path(ecfg["results_dir"]), city, 3)
        for sampler, _ in SAMPLERS:
            for seed in SEEDS:
                source = run_dir(ecfg, city, 3, sampler, seed)
                weights = source / "model.pt"
                with weights.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                if any(
                    info.get("source_checkpoint_sha256") == checksum and info["tag"] == LOO_TAG
                    and info["sampler"] == sampler and info["seed"] == seed
                    and info["evaluation"]["evaluation_id"] == evaluation["evaluation_id"]
                    and (path / "model.pt").is_file()
                    for path, info in prior_runs
                ):
                    continue
                saved_cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
                saved_eval = {k: v for k, v in saved_cfg["eval"].items() if k != "results_dir"}
                if saved_eval != {k: v for k, v in ecfg.items() if k != "results_dir"}:
                    raise ValueError(f"보관 체크포인트의 평가 설정이 다릅니다. {source}")
                source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
                scorer = SASRecScorer(load_trained(source).to(device), device, model_name(sampler))
                ks, batch = tuple(ecfg["ks"]), ecfg["batch_size"]
                valid = evaluate(scorer, split.valid, geo, cands["valid"], ks, batch)
                test = evaluate(scorer, split.test, geo, cands["test"], ks, batch).assign(model=scorer.name, seed=seed)
                test = report.with_meta(test, split.test)
                columns = ["query", "protocol", "user_idx", "is_repeat", "in_candidates"]
                original = pd.read_parquet(source / "per_query.parquet", columns=columns)
                if not original.sort_values(["query", "protocol"], ignore_index=True).equals(
                    test[columns].sort_values(["query", "protocol"], ignore_index=True)
                ):
                    raise ValueError(f"현재 R=3 쿼리가 보관 결과와 일치하지 않습니다. {source}")
                metrics = list(source_summary["test"])
                means = test.groupby("protocol")[metrics].mean().to_dict()
                out, record = new_run(
                    "sasrec", cfg, ecfg, evaluation, scfg={"model": saved_cfg["model"], "train": saved_cfg["train"]},
                    sampler=sampler, seed=seed, tag=LOO_TAG, device=str(device),
                )
                record.update(source_checkpoint=str(weights), source_checkpoint_sha256=checksum)
                primary, metric = ecfg["primary"]["protocol"], ecfg["primary"]["metric"]
                summary = {
                    **source_summary, "run_id": record["run_id"], "experiment_id": record["experiment_id"],
                    "tag": LOO_TAG, "device": str(device), "test": means,
                    f"best_valid_{primary}_{metric}": float(valid.loc[valid["protocol"] == primary, metric].mean()),
                    "archived_test_delta": {
                        name: {p: value - source_summary["test"][name][p] for p, value in table.items()}
                        for name, table in means.items()
                    },
                }
                with publish_directory(out) as staging:
                    write_json(staging / "config.json", saved_cfg)
                    write_json(staging / "summary.json", summary)
                    write_json(staging / "metadata.json", record)
                    copy2(weights, staging / "model.pt")
                    copy2(source / "train_log.csv", staging / "train_log.csv")
                    test.to_parquet(staging / "per_query.parquet", index=False)
                print(f"[refresh] {city} {sampler} seed={seed} | E4 {means[metric]['E4']:.4f} | {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="반경·샘플링·시간 분할 학습 실험")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--sasrec-config", default="configs/sasrec.yaml")
    parser.add_argument("--city", choices=CITIES)
    parser.add_argument("--refresh-r3", action="store_true", help="보관된 3km 체크포인트를 MPS로 다시 평가")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("이 실험은 MPS GPU에 접근할 수 있는 환경에서 실행합니다.")

    data = load_config(args.config)
    eval_cfg = load_config(args.eval_config)
    sasrec = load_config(args.sasrec_config)
    strategy = data.get("split", {}).get("strategy", "leave-one-out")
    if args.refresh_r3:
        if strategy != "leave-one-out":
            parser.error("--refresh-r3는 보관 체크포인트와 같은 leave-one-out 설정에서 사용합니다.")
        refresh_r3(data, eval_cfg, (args.city,) if args.city else CITIES)
        return
    tag = GLOBAL_TAG if strategy == "global-time" else LOO_TAG
    jobs = [job for job in training_jobs(strategy) if args.city is None or job.city == args.city]
    inputs = {}
    for city in dict.fromkeys(job.city for job in jobs):
        split, _, files = load_split_inputs({**data, "city": city})
        inputs[city] = fingerprint(files), split_fingerprint(split)
    for i, job in enumerate(jobs, 1):
        cfg = {**data, "city": job.city, "geo": {**data["geo"], "radius_km": float(job.radius)}}
        overrides = ["train.device=mps"]
        if job.sampler == "mixed":
            overrides.append(f"train.geo_ratio={job.alpha}")
        model_cfg = apply_overrides(sasrec, overrides)
        matches = [
            path for path, record in saved_runs(Path(eval_cfg["results_dir"]), job.city, job.radius)
            if record["tag"] == tag and record["sampler"] == job.sampler and record["seed"] == job.seed
            and record["device"] == "mps" and record["configs"] == {"data": cfg, "eval": eval_cfg, "sasrec": model_cfg}
            and (record["evaluation"]["data_id"], record["evaluation"]["split_id"]) == inputs[job.city]
            and (path / "model.pt").is_file()
        ]
        label = f"{job.city} R={job.radius} {job.sampler} α={job.alpha:g} seed={job.seed}"
        if matches:
            print(f"[matrix] {i}/{len(jobs)} 완료 실행 사용 | {label} | {matches[-1]}", flush=True)
            continue
        print(f"[matrix] {i}/{len(jobs)} 시작 | {label}", flush=True)
        result = train_run(cfg, eval_cfg, model_cfg, job.sampler, job.seed, tag, radius_km=job.radius)
        print(f"[matrix] {i}/{len(jobs)} 완료 | {result['run_dir']}", flush=True)
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
