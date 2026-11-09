"""SASRec 학습.

python -m dongne_rec.train.trainer --city NYC --sampler random --seed 0
valid 주 지표(E4 NDCG@10)로 early stopping 후 최고 체크포인트를 test로 평가해요.
결과: results/runs/{city}/R{r}/{run_id}/
  metadata.json, config.json, train_log.csv, per_query.parquet (test), summary.json, model.pt
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from dongne_rec.config import load_config, resolve_radius
from dongne_rec.data.geo import GeoIndex
from dongne_rec.data.split import Split
from dongne_rec.eval import report
from dongne_rec.eval.protocols import PROTOCOLS, evaluate
from dongne_rec.experiments import load_evaluation, new_run
from dongne_rec.models.sasrec import SASRec, SASRecScorer, pad_left
from dongne_rec.storage import publish_directory, write_json
from dongne_rec.train.samplers import SAMPLER_IDS, GeoSampler, MixedSampler, NegativeSampler, RandomSampler, user_item_matrix


def make_windows(train: list[np.ndarray], max_len: int, all_windows: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """학습 시퀀스 → (inputs, positives, users). inputs/positives는 (W, max_len) 왼쪽 패딩.

    all_windows=False면 원 SASRec처럼 유저당 마지막 max_len개만, True면 끝에서부터 max_len씩 잘라
    모든 위치를 한 번씩 정답으로 써요.
    """
    inputs, positives, users = [], [], []
    for u, s in enumerate(train):
        ends = range(len(s), 1, -max_len) if all_windows else [len(s)]
        for end in ends:
            chunk = s[max(0, end - max_len - 1) : end]
            if len(chunk) < 2:
                continue
            inputs.append(chunk[:-1])
            positives.append(chunk[1:])
            users.append(u)
    return pad_left(inputs, max_len), pad_left(positives, max_len), np.array(users, dtype=np.int64)


def make_sampler(name: str, split: Split, geo: GeoIndex, rng: np.random.Generator, tcfg: dict) -> NegativeSampler:
    user_items = user_item_matrix(split.train, split.n_items)
    if name == "random":
        return RandomSampler(split.n_items, user_items, rng)
    if name == "geo":
        return GeoSampler(split.n_items, user_items, rng, geo.indptr, geo.indices)
    if name == "mixed":
        return MixedSampler(split.n_items, user_items, rng, geo.indptr, geo.indices, geo_ratio=tcfg["geo_ratio"])
    raise ValueError(f"알 수 없는 sampler: {name}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def bce_loss(model: SASRec, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, int]:
    h = model.encode(seq)
    mask = pos > 0
    pos_logit = (h * model.item_emb(pos)).sum(-1)[mask]
    neg_logit = (h * model.item_emb(neg)).sum(-1)[mask]
    loss = F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit)) + F.binary_cross_entropy_with_logits(
        neg_logit, torch.zeros_like(neg_logit)
    )
    return loss, int(mask.sum())


def model_name(sampler: str) -> str:
    return f"SASRec-{SAMPLER_IDS[sampler]} ({sampler})"


def run_dir(ecfg: dict, city: str, radius: float, sampler: str, seed: int, tag: str | None = None) -> Path:
    """기존 형식으로 저장된 실행의 경로."""
    if tag:
        return Path(ecfg["results_dir"]) / "tuning" / "sasrec" / city / f"R{radius:g}" / f"{tag}_{sampler}_s{seed}"
    return Path(ecfg["results_dir"]) / "sasrec" / city / f"R{radius:g}" / f"{sampler}_s{seed}"


def apply_overrides(cfg: dict, assignments: list[str]) -> dict:
    """["model.dropout=0.5", ...] → 중첩 키를 YAML 값으로 덮어쓴 사본."""
    cfg = copy.deepcopy(cfg)
    for a in assignments:
        path, value = a.split("=", 1)
        *parents, leaf = path.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"config에 없는 키: {path}")
        node[leaf] = yaml.safe_load(value)
    return cfg


def train_run(
    cfg: dict, ecfg: dict, scfg: dict, sampler_name: str, seed: int, tag: str | None = None,
    *, radius_km: float | None = None,
) -> dict:
    mcfg, tcfg = scfg["model"], scfg["train"]
    radius = resolve_radius(cfg, radius_km)
    tag = tag or None
    ks, metric, primary = tuple(ecfg["ks"]), ecfg["primary"]["metric"], ecfg["primary"]["protocol"]
    set_seed(seed)

    split, geo, cands, evaluation = load_evaluation(cfg, ecfg, radius)
    inputs, positives, users = make_windows(split.train, mcfg["max_len"], tcfg["windows"] == "all")
    rng = np.random.default_rng(seed)
    sampler = make_sampler(sampler_name, split, geo, rng, tcfg)

    device = resolve_device(tcfg["device"])
    out, metadata = new_run(
        "sasrec", cfg, ecfg, evaluation, scfg=scfg, sampler=sampler_name, seed=seed, tag=tag, device=str(device),
    )
    model = SASRec(split.n_items, **mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.98))
    scorer = SASRecScorer(model, device, model_name(sampler_name))
    print(f"[train] {cfg['city']} {scorer.name} seed={seed} device={device} windows={len(inputs):,} params={sum(p.numel() for p in model.parameters()):,}")

    best, best_epoch, best_state, bad_evals, log = -np.inf, 0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, total, count = time.perf_counter(), 0.0, 0
        order = rng.permutation(len(inputs))
        for start in range(0, len(order), tcfg["batch_size"]):
            idx = order[start : start + tcfg["batch_size"]]
            negs = sampler.sample(inputs[idx], positives[idx], users[idx])
            seq, pos, neg = (torch.from_numpy(a).to(device) for a in (inputs[idx], positives[idx], negs))
            loss, n = bce_loss(model, seq, pos, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * n
            count += n
        row = {"epoch": epoch, "loss": total / count, "sec": time.perf_counter() - t0}

        if epoch % tcfg["eval_every"] == 0 or epoch == tcfg["epochs"]:
            res = evaluate(scorer, split.valid, geo, cands["valid"], ks, ecfg["batch_size"])
            means = res.groupby("protocol")[metric].mean()
            row.update({f"valid_{p}": means[p] for p in PROTOCOLS})
            improved = means[primary] > best
            if improved:
                best, best_epoch, bad_evals = means[primary], epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad_evals += 1
            print(f"[train] epoch {epoch:3d} loss {row['loss']:.4f} valid {metric} "
                  + " ".join(f"{p}={means[p]:.4f}" for p in PROTOCOLS) + (" *" if improved else ""))
            if bad_evals >= tcfg["patience"]:
                log.append(row)
                break
        log.append(row)
    train_sec = time.perf_counter() - started

    model.load_state_dict(best_state)
    per_query = evaluate(scorer, split.test, geo, cands["test"], ks, ecfg["batch_size"]).assign(model=scorer.name, seed=seed)
    per_query = report.with_meta(per_query, split.test)
    test_means = {m: per_query.groupby("protocol")[m].mean()[list(PROTOCOLS)].to_dict() for m in (*(f"{x}@{k}" for x in ("ndcg", "hr") for k in ks), "mrr")}

    summary = {
        "city": cfg["city"], "radius_km": radius, "model": scorer.name, "sampler": sampler_name, "seed": seed, "tag": tag,
        "run_id": metadata["run_id"], "experiment_id": metadata["experiment_id"],
        "device": str(device), "best_epoch": best_epoch, f"best_valid_{primary}_{metric}": best,
        "train_sec": train_sec, "test": test_means,
    }
    with publish_directory(out) as staging:
        write_json(staging / "config.json", {"model": mcfg, "train": tcfg, "eval": ecfg})
        write_json(staging / "summary.json", summary)
        write_json(staging / "metadata.json", metadata)
        pd.DataFrame(log).to_csv(staging / "train_log.csv", index=False)
        per_query.to_parquet(staging / "per_query.parquet", index=False)
        torch.save(best_state, staging / "model.pt")
    print(f"[train] best epoch {best_epoch} ({train_sec:.0f}s) test {metric} "
          + " ".join(f"{p}={test_means[metric][p]:.4f}" for p in PROTOCOLS) + f" → {out}")
    return {**summary, "run_dir": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="SASRec 학습")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--sasrec-config", default="configs/sasrec.yaml")
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--sampler", default="random", choices=list(SAMPLER_IDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="sasrec config 덮어쓰기 (예: model.dropout=0.5)")
    parser.add_argument("--tag", help="튜닝 실행의 이름. 기본 비교 목록에서 제외하며 --run으로 선택할 수 있습니다.")
    args = parser.parse_args()

    scfg = apply_overrides(load_config(args.sasrec_config), args.set)
    train_run(
        load_config(args.config, city=args.city), load_config(args.eval_config), scfg, args.sampler, args.seed, args.tag,
        radius_km=args.radius_km,
    )


if __name__ == "__main__":
    main()
