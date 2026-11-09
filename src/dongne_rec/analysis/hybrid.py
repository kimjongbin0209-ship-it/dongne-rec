"""Tune a history-aware rank fusion on valid and evaluate frozen weights on test."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dongne_rec.baselines.hybrid import RRF_OFFSET, fuse_ranks, reciprocal_ranks
from dongne_rec.baselines.pop import GeoPop, PersonalFreq
from dongne_rec.config import load_config
from dongne_rec.data.geo import GeoIndex
from dongne_rec.data.split import Queries, Split
from dongne_rec.eval import report
from dongne_rec.eval.metrics import expected_metrics
from dongne_rec.eval.stats import paired_bootstrap
from dongne_rec.experiments import attempt_id, environment_metadata, git_metadata, load_evaluation
from dongne_rec.models.sasrec import SASRecScorer
from dongne_rec.serving.export_onnx import load_trained
from dongne_rec.storage import publish_directory, write_json
from dongne_rec.train.trainer import model_name, resolve_device

CITIES = ("NYC", "TKY")
VARIANTS = {"S0": "random", "S2": "mixed"}
SEEDS = (0, 1, 2)
RADIUS_KM = 3
K = 10
WEIGHTS = np.array(list(itertools.product(np.linspace(0, 1, 11), repeat=2))).round(1)
CONTROLS = ("GeoPop", "PersonalFreq", "SASRec-S0", "SASRec-S2")


@dataclass
class Pool:
    queries: Queries
    items: list[np.ndarray]
    visited: list[np.ndarray]
    targets: np.ndarray


def e4_pool(queries: Queries, geo: GeoIndex) -> Pool:
    items, visited = [], []
    targets = np.full(len(queries), -1, dtype=np.int64)
    for i, (history, last, target) in enumerate(zip(queries.histories, queries.last_items, queries.targets)):
        candidates = geo.neighbors(last)
        candidates = candidates[(candidates != last) & (candidates != 0)]
        items.append(candidates)
        visited.append(np.isin(candidates, history))
        found = np.flatnonzero(candidates == target)
        if found.size:
            targets[i] = found[0]
    return Pool(queries, items, visited, targets)


def local_scores(scorer, pool: Pool, batch_size: int) -> list[np.ndarray]:
    q = pool.queries
    local = []
    for start in range(0, len(q), batch_size):
        stop = min(start + batch_size, len(q))
        scores = scorer.score(q.histories[start:stop], q.last_items[start:stop])
        local.extend(row[ids] for row, ids in zip(scores, pool.items[start:stop]))
    return local


def evaluate_e4(pool: Pool, scores: list[np.ndarray]) -> pd.DataFrame:
    greater, ties = np.zeros(len(pool.targets), dtype=np.int64), np.zeros(len(pool.targets), dtype=np.int64)
    for i, (row, target) in enumerate(zip(scores, pool.targets)):
        if target >= 0:
            greater[i] = np.count_nonzero(row > row[target])
            ties[i] = np.count_nonzero(row == row[target]) - 1
    found = pool.targets >= 0
    return report.with_meta(pd.DataFrame({
        "query": np.arange(len(pool.targets)), "protocol": "E4", "in_candidates": found,
        **{key: np.where(found, values, 0.0) for key, values in expected_metrics(greater, ties, (K,)).items()},
    }), pool.queries)


def grid_e4(pool: Pool, personal: list[np.ndarray], sequential: list[np.ndarray]) -> np.ndarray:
    total = np.zeros(len(WEIGHTS))
    for pf, sas, visited, target in zip(personal, sequential, pool.visited, pool.targets):
        if target < 0:
            continue
        scores = fuse_ranks(pf, sas, visited, WEIGHTS)
        target_scores = scores[:, target, None]
        greater = (scores > target_scores).sum(axis=1)
        ties = (scores == target_scores).sum(axis=1) - 1
        total += expected_metrics(greater, ties, (K,))[f"ndcg@{K}"]
    return total / len(pool.targets)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    run_id: str
    sha256: str

    def scorer(self, device: torch.device, sampler: str) -> SASRecScorer:
        if checkpoint_hash(self.path) != self.sha256:
            raise ValueError(f"가중치 선택 이후 체크포인트가 바뀌었습니다. {self.path}")
        return SASRecScorer(load_trained(self.path).to(device), device, model_name(sampler))


def checkpoint_hash(run: Path) -> str:
    with (run / "model.pt").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Group:
    city: str
    split: Split
    geo: GeoIndex
    evaluation: dict
    baseline: Path
    checkpoints: dict[tuple[str, int], Checkpoint]

    def context(self) -> dict:
        return {"city": self.city, "split": self.split.strategy}


def load_group(data: dict, ecfg: dict, city: str, sources: list[dict]) -> Group:
    split, geo, _, evaluation = load_evaluation({**data, "city": city}, ecfg, RADIUS_KM)
    baseline = None
    checkpoints = {}
    for source in sources:
        if (source["city"], source["split"], source["radius_km"]) != (city, split.strategy, RADIUS_KM):
            continue
        path = Path(source["path"])
        record = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if record["run_id"] != source["run_id"] or record["evaluation"]["evaluation_id"] != evaluation["evaluation_id"]:
            raise ValueError(f"3단계 실행과 현재 평가 지문이 일치하지 않습니다. {path}")
        if record["kind"] == "baselines":
            if baseline is not None:
                raise ValueError(f"베이스라인 실행이 중복됩니다. {city} {split.strategy}")
            baseline = path
        elif record["sampler"] in VARIANTS.values() and record["seed"] in SEEDS:
            if record["sampler"] == "mixed" and record["configs"]["sasrec"]["train"]["geo_ratio"] != 0.5:
                continue
            key = (record["sampler"], record["seed"])
            if key in checkpoints:
                raise ValueError(f"체크포인트가 중복됩니다. {path}")
            checkpoints[key] = Checkpoint(path, record["run_id"], checkpoint_hash(path))
    expected = set(itertools.product(VARIANTS.values(), SEEDS))
    if baseline is None or set(checkpoints) != expected:
        raise ValueError(f"{city} {split.strategy}에 필요한 3km 대조군이 없습니다. {expected - checkpoints.keys()}")
    return Group(city, split, geo, evaluation, baseline, checkpoints)


def tune(group: Group, device: torch.device, batch_size: int) -> tuple[dict, list[dict]]:
    pool = e4_pool(group.split.valid, group.geo)
    personal = PersonalFreq().fit(group.split.train, group.split.n_items)
    pf = [reciprocal_ranks(scores) for scores in local_scores(personal, pool, batch_size)]
    chosen, rows = {}, []
    for variant, sampler in VARIANTS.items():
        per_seed = []
        for seed in SEEDS:
            scorer = group.checkpoints[(sampler, seed)].scorer(device, sampler)
            ranks = [reciprocal_ranks(scores) for scores in local_scores(scorer, pool, batch_size)]
            per_seed.append(grid_e4(pool, pf, ranks))
            del scorer
        means = np.mean(per_seed, axis=0)
        # Exact ties prefer fewer history-dependent differences, then a higher PersonalFreq weight.
        index = max(range(len(means)), key=lambda i: (means[i], -abs(WEIGHTS[i, 0] - WEIGHTS[i, 1]), *WEIGHTS[i]))
        chosen[variant] = {
            "pf_visited": float(WEIGHTS[index, 0]), "pf_unvisited": float(WEIGHTS[index, 1]),
            "valid_ndcg@10": float(means[index]),
        }
        for i, (seen, new) in enumerate(WEIGHTS):
            rows.append({
                **group.context(), "variant": variant, "pf_visited": seen, "pf_unvisited": new,
                "valid_ndcg@10": means[i], "selected": i == index,
                **{f"seed_{seed}": values[i] for seed, values in zip(SEEDS, per_seed)},
            })
    return chosen, rows


def exposure(pool: Pool, scores: list[np.ndarray]) -> dict:
    unique = set()
    total = new = 0
    for ids, visited, row in zip(pool.items, pool.visited, scores):
        top = np.lexsort((ids, -row))[:K]
        unique.update(ids[top].tolist())
        new += int(np.count_nonzero(~visited[top]))
        total += len(top)
    return {"unique_items": len(unique), "unvisited_share": new / total if total else 0.0}


def reference_delta(measured: pd.DataFrame, source: Path, model: str) -> float:
    archived = pd.read_parquet(source / "per_query.parquet")
    archived = archived[(archived["model"] == model) & (archived["protocol"] == "E4")].sort_values("query")
    columns = ["query", "user_idx", "is_repeat", "in_candidates"]
    if not np.array_equal(measured[columns].to_numpy(), archived[columns].to_numpy()):
        raise ValueError(f"3단계 E4 쿼리와 현재 쿼리가 일치하지 않습니다. {source} {model}")
    metrics = ["ndcg@10", "hr@10"]
    return float(np.max(np.abs(measured[metrics].to_numpy() - archived[metrics].to_numpy())))


def test_group(
    group: Group, chosen: dict, device: torch.device, batch_size: int,
) -> tuple[list[pd.DataFrame], list[dict], list[dict]]:
    pool = e4_pool(group.split.test, group.geo)
    frames, exposures, checks = [], [], []

    def save(model: str, seed: int, scores: list[np.ndarray], reference: Path | None = None, source_name: str = "") -> None:
        frame = evaluate_e4(pool, scores)
        keys = {**group.context(), "model": model, "seed": seed}
        frames.append(frame.assign(**keys))
        exposures.append({**keys, **exposure(pool, scores)})
        if reference is not None:
            checks.append({**keys, "max_query_metric_delta": reference_delta(frame, reference, source_name)})

    geo = GeoPop(group.geo).fit(group.split.train, group.split.n_items)
    save("GeoPop", -1, local_scores(geo, pool, batch_size), group.baseline, geo.name)
    personal = PersonalFreq().fit(group.split.train, group.split.n_items)
    personal_scores = local_scores(personal, pool, batch_size)
    save("PersonalFreq", -1, personal_scores, group.baseline, personal.name)
    pf = [reciprocal_ranks(scores) for scores in personal_scores]
    for variant, sampler in VARIANTS.items():
        weights = np.array([chosen[variant]["pf_visited"], chosen[variant]["pf_unvisited"]])
        for seed in SEEDS:
            checkpoint = group.checkpoints[(sampler, seed)]
            scorer = checkpoint.scorer(device, sampler)
            sequential = local_scores(scorer, pool, batch_size)
            save(f"SASRec-{variant}", seed, sequential, checkpoint.path, scorer.name)
            combined = [
                fuse_ranks(p, reciprocal_ranks(s), visited, weights)
                for p, s, visited in zip(pf, sequential, pool.visited)
            ]
            save(f"Hybrid-{variant}", seed, combined)
            del scorer
    return frames, exposures, checks


def summarize(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, effects = [], []
    for (city, split), frame in test.groupby(["city", "split"], sort=False):
        masks = {
            "all": np.ones(len(frame), dtype=bool),
            "repeat": frame["is_repeat"],
            "new": ~frame["is_repeat"],
            "known-new": ~frame["is_repeat"] & ~frame["is_cold"],
        }
        for scope, mask in masks.items():
            rows = frame.loc[mask]
            if rows.empty:
                continue
            keys = {"city": city, "split": split, "slice": scope, "queries": rows["query"].nunique()}
            seed_means = rows.groupby(["model", "seed"])[["ndcg@10", "hr@10"]].mean()
            for model, means in seed_means.groupby(level="model", sort=False):
                metrics.append({
                    **keys, "model": model, "ndcg": means["ndcg@10"].mean(),
                    "ndcg_sd": means["ndcg@10"].std(), "hr": means["hr@10"].mean(),
                })
            paired = rows.groupby(["query", "model"])["ndcg@10"].mean().unstack("model")
            for variant, control in itertools.product(VARIANTS, CONTROLS):
                model = f"Hybrid-{variant}"
                effects.append({
                    **keys, "model": model, "control": control,
                    **paired_bootstrap(paired[control].to_numpy(), paired[model].to_numpy()),
                })
    return pd.DataFrame(metrics), pd.DataFrame(effects)


def display_score(row: pd.Series) -> str:
    if pd.isna(row["ndcg"]):
        return "—"
    score = f"{row['ndcg']:.4f}"
    return score if pd.isna(row["ndcg_sd"]) else f"{score} ± {row['ndcg_sd']:.4f}"


def render(matrix: pd.DataFrame, effects: pd.DataFrame, exposures: pd.DataFrame, selection: dict, checks: list[dict]) -> str:
    lines = [
        "# 방문 이력 기반 PersonalFreq–SASRec 순위 결합",
        "",
        "뉴욕과 도쿄에서 3km E4를 평가했습니다. 각 요청의 반경 안에서 현재 장소를 빼고 PersonalFreq와 SASRec의 순위를 구했습니다. "
        f"모델별 동점은 평균 순위로 바꾸고, `1/({RRF_OFFSET}+순위)`를 가중합했습니다.",
        "",
        "요청 시점까지 방문한 후보와 미방문 후보에는 서로 다른 PersonalFreq 가중치를 적용했습니다. "
        "각 그룹에서 두 가중치를 0.0~1.0까지 0.1 간격으로 조합했습니다. "
        "시드 0·1·2의 valid NDCG@10 평균이 가장 높은 쌍을 고정한 뒤 test를 평가했습니다.",
        "",
        f"3단계의 학습 체크포인트를 사용해 {selection['device'].upper()}로 추론했습니다. "
        f"단독 모델의 3단계 E4 대비 쿼리별 지표 최대 절대 차이는 {max(row['max_query_metric_delta'] for row in checks):.3g}입니다.",
        "",
        "## valid에서 고른 가중치",
        "",
    ]
    weight_rows = {}
    for group in selection["groups"]:
        for variant, choice in group["weights"].items():
            weight_rows[f"{group['city']} · {group['split']} · {variant}"] = {
                "방문 후보 PF": f"{choice['pf_visited']:.1f}",
                "미방문 후보 PF": f"{choice['pf_unvisited']:.1f}",
                "valid E4": f"{choice['valid_ndcg@10']:.4f}",
            }
    lines.extend([report.md_table(pd.DataFrame.from_dict(weight_rows, orient="index"), fmt="{}", index_name="도시 · 분할 · 모델"), ""])
    lines.extend([
        "## test E4와 상위 10개 노출",
        "",
        "표의 점수는 NDCG@10입니다. ±는 시드별 평균의 표준편차입니다. E4는 반경 밖과 카탈로그 밖 정답을 miss로 계산합니다. "
        "전역 시간 분할의 등록 신규는 학습 카탈로그에 있는 미방문 정답입니다.",
        "",
        "미방문 노출 비중과 고유 장소 수는 실제 상위 10개 목록으로 계산했습니다. "
        "노출 계산에서 동점은 장소 ID 순으로 정렬하고, E4는 동점 순위의 기댓값을 사용합니다.",
        "",
    ])
    for (city, split), group in matrix.groupby(["city", "split"], sort=False):
        indexed = group.set_index(["model", "slice"])
        shown = exposures[(exposures["city"] == city) & (exposures["split"] == split)]
        exposed = shown.groupby("model")[["unique_items", "unvisited_share"]].mean()
        scopes = {"all": "전체", "repeat": "재방문", "new": "신규"}
        if split == "global-time":
            scopes["known-new"] = "등록 신규"
        rows = {}
        for model in (*CONTROLS, *(f"Hybrid-{v}" for v in VARIANTS)):
            available = indexed.loc[model].reindex(list(scopes))
            scores = {label: display_score(available.loc[scope]) for scope, label in scopes.items()}
            rows[model] = {
                **scores, "전체 HR@10": f"{indexed.loc[(model, 'all'), 'hr']:.4f}",
                "미방문 노출": f"{exposed.loc[model, 'unvisited_share']:.1%}",
                "고유 장소": f"{exposed.loc[model, 'unique_items']:,.0f}",
            }
        counts = ", ".join(
            f"{label} {int(indexed['queries'].get(('PersonalFreq', scope), 0)):,}건" for scope, label in scopes.items()
        )
        lines.extend([f"### {city} · {split}", "", counts, "", report.md_table(pd.DataFrame.from_dict(rows, orient="index"), fmt="{}"), ""])
    lines.extend([
        "## 혼합 모델의 차이와 신뢰구간",
        "",
        "같은 요청의 세 시드 점수를 평균한 뒤 이용자 쿼리를 2,000회 재표집했습니다. "
        "괄호는 다중 비교 보정 전 95% 신뢰구간입니다.",
        "",
    ])
    indexed = effects.set_index(["city", "split", "model", "slice", "control"])

    def diff(city: str, split: str, model: str, scope: str, control: str) -> str:
        key = (city, split, model, scope, control)
        if key not in indexed.index:
            return "—"
        row = indexed.loc[key]
        effect, low, high = [
            "0" if x == 0 else f"{x:+.2e}" if abs(x) < 0.00005 else f"{x:+.4f}"
            for x in (row["diff"], row["lo"], row["hi"])
        ]
        return f"{effect} [{low}, {high}]"

    delta_rows = {}
    for group in selection["groups"]:
        city, split = group["city"], group["split"]
        for variant in VARIANTS:
            model = f"Hybrid-{variant}"
            delta_rows[f"{city} · {split} · {model}"] = {
                "전체 − PF": diff(city, split, model, "all", "PersonalFreq"),
                "재방문 − PF": diff(city, split, model, "repeat", "PersonalFreq"),
                "신규 − SASRec": diff(city, split, model, "new", f"SASRec-{variant}"),
                "신규 − GeoPop": diff(city, split, model, "new", "GeoPop"),
            }
    lines.extend([
        report.md_table(pd.DataFrame.from_dict(delta_rows, orient="index"), fmt="{}", index_name="도시 · 분할 · 모델"),
        "",
        "실행 근거는 [고정 가중치와 체크포인트](selection.json), [valid 탐색표](valid_grid.csv), [test 지표](matrix.csv), "
        "[전체 대조군 신뢰구간](paired_bootstrap.csv), [쿼리별 결과](test_queries.parquet)에 저장했습니다.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="valid에서 이력별 순위 가중치를 선택하고 E4 test 평가")
    parser.add_argument("--stage3-inputs", type=Path, required=True)
    parser.add_argument("--loo-config", default="configs/data.yaml")
    parser.add_argument("--global-config", default="configs/data_global.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS GPU에 접근할 수 있는 환경에서 실행하세요.")
    sources = json.loads(args.stage3_inputs.read_text(encoding="utf-8"))["runs"]
    ecfg = load_config(args.eval_config)
    batch = ecfg["batch_size"]
    groups = [
        load_group(data, ecfg, city, sources)
        for data in map(load_config, (args.loo_config, args.global_config))
        for city in CITIES
    ]
    out = Path(ecfg["results_dir"]) / "reports" / "hybrid" / attempt_id("stage4")
    with publish_directory(out) as staging:
        selected, grid = [], []
        for group in groups:
            print(f"[hybrid] {group.city} {group.split.strategy} valid", flush=True)
            chosen, rows = tune(group, device, batch)
            selected.append({
                **group.context(), "evaluation": group.evaluation,
                "valid_queries": len(group.split.valid), "weights": chosen, "baseline": str(group.baseline),
                "checkpoints": [
                    {"variant": variant, "seed": seed, "path": str(group.checkpoints[(sampler, seed)].path),
                     "run_id": group.checkpoints[(sampler, seed)].run_id, "sha256": group.checkpoints[(sampler, seed)].sha256}
                    for variant, sampler in VARIANTS.items() for seed in SEEDS
                ],
            })
            grid.extend(rows)
        selection = {
            "selection_stage": "valid", "metric": "E4 NDCG@10", "radius_km": RADIUS_KM, "rrf_offset": RRF_OFFSET,
            "seeds": list(SEEDS), "device": str(device), "stage3_inputs": str(args.stage3_inputs), "groups": selected,
            "environment": environment_metadata(), "git": git_metadata(),
        }
        write_json(staging / "selection.json", selection)
        pd.DataFrame(grid).to_csv(staging / "valid_grid.csv", index=False)

        frames, exposures, checks = [], [], []
        for group, entry in zip(groups, selected):
            print(f"[hybrid] {group.city} {group.split.strategy} test with frozen weights", flush=True)
            scores, impressions, reference = test_group(group, entry["weights"], device, batch)
            frames.extend(scores)
            exposures.extend(impressions)
            checks.extend(reference)
        test = pd.concat(frames, ignore_index=True)
        exposed = pd.DataFrame(exposures)
        matrix, effects = summarize(test)
        test.to_parquet(staging / "test_queries.parquet", index=False)
        matrix.to_csv(staging / "matrix.csv", index=False)
        effects.to_csv(staging / "paired_bootstrap.csv", index=False)
        exposed.to_csv(staging / "exposure.csv", index=False)
        pd.DataFrame(checks).to_csv(staging / "reference_checks.csv", index=False)
        (staging / "report.md").write_text(render(matrix, effects, exposed, selection, checks), encoding="utf-8")
    print(f"[hybrid] → {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
