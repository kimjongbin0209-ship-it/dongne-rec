"""Aggregate the completed stage 3 runs, with the exact input runs recorded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from dongne_rec.analysis.sensitivity import CITIES, GLOBAL_TAG, LOO_TAG, SAMPLERS, SEEDS
from dongne_rec.config import load_config
from dongne_rec.data.geo import load_geo_index
from dongne_rec.data.split import load_split_inputs, split_fingerprint
from dongne_rec.eval import figures
from dongne_rec.eval.compare import S0, S1, S2, load_results
from dongne_rec.eval.protocols import PROTOCOLS
from dongne_rec.eval.stats import paired_bootstrap
from dongne_rec.experiments import attempt_id, saved_runs
from dongne_rec.storage import fingerprint, publish_directory, write_json

BASE_MODELS = ("MostPop", "GeoPop", "PersonalFreq", "ItemKNN", "S0", "S1", "S2")
ALPHAS = ((0.0, "S0"), (0.25, "S2 α=0.25"), (0.5, "S2"), (0.75, "S2 α=0.75"), (1.0, "S1"))
LABELS = {S0: "S0", S1: "S1", S2: "S2", f"{S2} α=0.25": "S2 α=0.25", f"{S2} α=0.75": "S2 α=0.75"}


def load_group(data: dict, ecfg: dict, scfg: dict, city: str, radius: int) -> tuple[pd.DataFrame, dict, dict, list[dict]]:
    cfg = {**data, "city": city, "geo": {**data["geo"], "radius_km": float(radius)}}
    split, items, files = load_split_inputs(cfg)
    data_id, split_id = fingerprint(files), split_fingerprint(split)
    tag = GLOBAL_TAG if split.strategy == "global-time" else LOO_TAG
    conditions = list(SAMPLERS)
    if tag == LOO_TAG and radius == 3:
        conditions.extend(("mixed", alpha) for alpha in (0.25, 0.75))
    expected = [(sampler, alpha, seed) for sampler, alpha in conditions for seed in SEEDS]
    baseline = None
    trained = {}
    for path, record in saved_runs(Path(ecfg["results_dir"]), city, radius):
        if record["tag"] != tag or record["configs"]["data"] != cfg or record["configs"]["eval"] != ecfg:
            continue
        evaluation = record["evaluation"]
        if (evaluation["data_id"], evaluation["split_id"]) != (data_id, split_id):
            continue
        if record["kind"] == "baselines":
            baseline = (path, record)
        elif record["kind"] == "sasrec" and record["device"] == "mps":
            actual = record["configs"]["sasrec"]
            sampler = record["sampler"]
            alpha = actual["train"]["geo_ratio"] if sampler == "mixed" else dict(SAMPLERS)[sampler]
            train = {k: v for k, v in actual["train"].items() if k not in {"device", "geo_ratio"}}
            requested = {k: v for k, v in scfg["train"].items() if k not in {"device", "geo_ratio"}}
            if actual["model"] == scfg["model"] and train == requested:
                trained[(sampler, alpha, record["seed"])] = (path, record)
    missing = [key for key in expected if key not in trained]
    if baseline is None or missing:
        raise ValueError(f"{city} {split.strategy} R={radius}에서 필요한 실행이 없습니다. baseline={baseline is not None}, {missing}")
    selected = [baseline, *(trained[key] for key in expected)]
    if len({r["evaluation"]["evaluation_id"] for _, r in selected}) != 1:
        raise ValueError(f"{city} {split.strategy} R={radius}의 후보 캐시가 일치하지 않습니다.")

    # The group tag identifies the experiment batch, not a separate model.
    query = load_results([(path, {**r, "tag": None}) for path, r in selected])
    query["model"] = query["model"].map(lambda m: "ItemKNN" if m.startswith("ItemKNN(") else LABELS.get(m, m))
    query["is_cold"] = query["is_cold"].eq(True) if "is_cold" in query else False
    valid = {}
    inputs = []
    for path, record in selected:
        s = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        inputs.append({
            "path": str(path), "run_id": record["run_id"], "city": city, "radius_km": radius,
            "split": split.strategy, "evaluation_id": record["evaluation"]["evaluation_id"],
            "kind": record["kind"], "device": record["device"], "sampler": record["sampler"],
            "seed": record["seed"], "source_checkpoint": record.get("source_checkpoint"),
            "archived_test_delta": s.get("archived_test_delta"),
        })
        if record["kind"] == "sasrec":
            name = s["model"]
            if record["sampler"] == "mixed":
                ratio = record["configs"]["sasrec"]["train"]["geo_ratio"]
                if ratio != 0.5:
                    name += f" α={ratio:g}"
            valid.setdefault(LABELS[name], []).append(s["best_valid_E4_ndcg@10"])
    reference = query[(query["model"] == "PersonalFreq") & (query["protocol"] == "E4")]
    geo = load_geo_index(cfg, radius, items=items)
    pool_sizes = np.diff(geo.indptr)[split.test.last_items] - 1
    cohort = {
        "split": split.strategy, "city": city, "radius_km": radius,
        "train_users": len(split.train), "train_checkins": sum(map(len, split.train)), "catalog_items": len(items),
        "valid_queries": len(split.valid), "test_queries": len(split.test), "cold_queries": int(reference["is_cold"].sum()),
        "repeat_share": float(reference["is_repeat"].mean()), "coverage": float(reference["in_candidates"].mean()),
        "median_candidates": float(np.median(pool_sizes)),
    }
    return query, cohort, valid, inputs


def summarize(query: pd.DataFrame, cohort: dict, valid: dict) -> tuple[list[dict], list[dict]]:
    keys = {name: cohort[name] for name in ("split", "city", "radius_km")}
    rows = []
    for model, frame in query.groupby("model", sort=False):
        seeds = frame.groupby(["seed", "protocol"])[["ndcg@10", "hr@10"]].mean()
        average = seeds.groupby("protocol").mean()
        e4 = frame[frame["protocol"] == "E4"]
        valid_scores = valid.get(model)
        rows.append({
            **keys, "model": model, "seeds": len(valid_scores) if valid_scores else 0,
            "valid_e4": float(np.mean(valid_scores)) if valid_scores else np.nan,
            **{f"{p.lower()}_ndcg": float(average.loc[p, "ndcg@10"]) for p in PROTOCOLS},
            "e4_sd": float(seeds.xs("E4", level="protocol")["ndcg@10"].std()),
            "e4_hr": float(average.loc["E4", "hr@10"]),
            "e4_repeat": float(e4.loc[e4["is_repeat"], "ndcg@10"].mean()),
            "e4_new": float(e4.loc[~e4["is_repeat"], "ndcg@10"].mean()),
            "e4_warm_new": float(e4.loc[~e4["is_repeat"] & ~e4["is_cold"], "ndcg@10"].mean()),
            "e4_in_radius": float(e4.loc[e4["in_candidates"], "ndcg@10"].mean()),
        })
    e4 = query[query["protocol"] == "E4"]
    slices = {
        "E3": query[query["protocol"] == "E3"], "E4": e4, "E4 repeat": e4[e4["is_repeat"]],
        "E4 new": e4[~e4["is_repeat"]], "E4 warm-new": e4[~e4["is_repeat"] & ~e4["is_cold"]],
        "E4 in-radius": e4[e4["in_candidates"]],
    }
    effects = []
    for label, frame in slices.items():
        if frame.empty:
            continue
        scores = frame.groupby(["query", "model"])["ndcg@10"].mean().unstack("model")
        pairs = [("S0", m) for m in ("S1", "S2", "S2 α=0.25", "S2 α=0.75")]
        if label == "E4":
            pairs.extend(("S2", m) for m in ("S2 α=0.25", "S2 α=0.75"))
        for base, model in pairs:
            if base in scores and model in scores:
                effects.append({
                    **keys, "slice": label, "model": model, "baseline": base, "queries": len(scores),
                    **paired_bootstrap(scores[base].to_numpy(), scores[model].to_numpy()),
                })
    return rows, effects


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|---" * len(headers) + "|",
            *["| " + " | ".join(row) + " |" for row in rows], ""]


def score(value: float, sd: float = np.nan) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.4f}" if pd.isna(sd) else f"{value:.4f} ± {sd:.4f}"


def render(scores: pd.DataFrame, cohorts: pd.DataFrame, effects: pd.DataFrame, inputs: list[dict]) -> str:
    loo = scores[scores["split"] == "leave-one-out"]
    radius_rows, near_rows, alpha_rows, selected_rows, time_rows, model_rows, effect_rows = [], [], [], [], [], [], []
    for city in CITIES:
        standard = loo[(loo["city"] == city) & loo["model"].isin(BASE_MODELS)]
        for radius in (2, 3, 5):
            models = standard[standard["radius_km"] == radius].set_index("model")
            cohort = cohorts.loc[(city, "leave-one-out", radius)]
            radius_rows.append([
                city, str(radius), f"{cohort['coverage']:.1%}", f"{cohort['median_candidates']:,.0f}",
                score(models.loc["PersonalFreq", "e4_ndcg"]),
                *[score(models.loc[m, "e4_ndcg"], models.loc[m, "e4_sd"]) for m in ("S0", "S1", "S2")],
                str(models["e1_ndcg"].idxmax()), str(models["e4_ndcg"].idxmax()),
            ])
            near_rows.append([
                city, str(radius), f"{round(cohort['coverage'] * cohort['test_queries']):,}",
                *[score(models.loc[m, "e4_in_radius"]) for m in ("PersonalFreq", "S0", "S1", "S2")],
            ])
        for model in ("S0", "S1", "S2"):
            best = standard[standard["model"] == model].sort_values(["valid_e4", "radius_km"], ascending=[False, True]).iloc[0]
            selected_rows.append([city, model, str(best["radius_km"]), score(best["valid_e4"]), score(best["e4_ndcg"])])
        r3 = loo[(loo["city"] == city) & (loo["radius_km"] == 3)].set_index("model")
        winner = r3.loc[[m for _, m in ALPHAS], "valid_e4"].idxmax()
        for alpha, model in ALPHAS:
            r = r3.loc[model]
            alpha_rows.append([
                city, f"**{alpha:g}**" if model == winner else f"{alpha:g}", score(r["valid_e4"]),
                score(r["e1_ndcg"]), score(r["e3_ndcg"]), score(r["e4_ndcg"], r["e4_sd"]), score(r["e4_hr"]),
                score(r["e4_repeat"]), score(r["e4_new"]), score(r["e4_in_radius"]),
            ])
        for split in ("leave-one-out", "global-time"):
            group = scores[(scores["city"] == city) & (scores["split"] == split) & (scores["radius_km"] == 3)]
            group = group[group["model"].isin(BASE_MODELS)].set_index("model")
            c = cohorts.loc[(city, split, 3)]
            ranks = [str(group[f"e{p}_ndcg"].idxmax()) for p in (1, 3, 4)]
            tau = kendalltau(group["e1_ndcg"], group["e4_ndcg"]).statistic
            time_rows.append([
                city, split, f"{int(c['test_queries']):,}",
                f"{int(c['cold_queries']):,} ({c['cold_queries'] / c['test_queries']:.1%})",
                f"{c['coverage']:.1%}", f"{c['repeat_share']:.1%}", *ranks, f"{tau:.2f}",
            ])
            for model in ("PersonalFreq", "GeoPop", "S0", "S1", "S2"):
                r = group.loc[model]
                model_rows.append([
                    city, split, model, score(r["e1_ndcg"]), score(r["e3_ndcg"]), score(r["e4_ndcg"], r["e4_sd"]),
                    score(r["e4_hr"]), score(r["e4_repeat"]), score(r["e4_new"]), score(r["e4_warm_new"]),
                    score(r["e4_in_radius"]),
                ])
    for row in effects[effects["slice"].eq("E4")].itertuples():
        effect_rows.append([
            row.city, row.split, str(row.radius_km), f"{row.model} − {row.baseline}", f"{row.diff:+.4f}",
            f"[{row.lo:+.4f}, {row.hi:+.4f}]", f"{row.rel:+.1%}",
        ])
    archived = [r for r in inputs if r["source_checkpoint"]]
    trained = sum(r["kind"] == "sasrec" for r in inputs) - len(archived)
    archive_delta = max(
        (abs(x) for row in archived for x in row["archived_test_delta"]["ndcg@10"].values()),
        default=0,
    )
    return "\n".join([
        "# 반경·혼합 비율·전역 시간 분할 실험",
        "",
        f"시드 0·1·2로 SASRec을 MPS에서 {trained}회 학습했습니다. 3km 보관 체크포인트 {len(archived)}개는 "
        "현재 후보와 평가 코드로 MPS에서 다시 평가했습니다. 베이스라인은 CPU에서 평가했습니다.",
        f"보관된 지표와 재평가한 지표의 최대 NDCG@10 차이는 {archive_delta:.3g}이었습니다.",
        "",
        "체크포인트와 반경·α 선택에는 valid E4 NDCG@10을 사용했습니다. 표는 사전에 정한 모든 설정의 test 결과를 함께 보여줍니다. "
        "반경·α 민감도는 초기 leave-one-out test 결과를 확인한 뒤 진행한 탐색입니다. "
        "±는 세 시드 평균의 표본 표준편차입니다.",
        "",
        "전역 분할은 2012-12-01 00:00 UTC 전에 학습 카탈로그·좌표·k-core를 확정하고, 2013-01-01 00:00 UTC부터 test를 평가합니다. "
        "각 기간에는 학습 이용자당 첫 추천 가능 시점을 사용했습니다. 이력은 요청 전 등록 장소만 포함합니다. "
        "카탈로그에 없는 정답은 모든 프로토콜에서 miss입니다. 신규에는 미등록 정답을 포함하며, 등록 신규에는 이용자 미방문 등록 장소만 포함합니다.",
        "",
        "![반경과 혼합 비율에 따른 E4 NDCG@10](sensitivity.png)",
        "",
        "## 반경 비교 (leave-one-out)",
        "",
        *table(["도시", "km", "정답 포함률", "후보 중앙값", "PersonalFreq E4", "S0 E4", "S1 E4", "S2 E4", "E1 1위", "E4 1위"], radius_rows),
        "반경 내 정답만 남긴 E4입니다. 반경에 따라 이 표에 포함되는 쿼리가 달라집니다.",
        "",
        *table(["도시", "km", "반경 내 정답 수", "PersonalFreq", "S0", "S1", "S2"], near_rows),
        "### valid로 선택한 반경",
        "",
        *table(["도시", "샘플러", "선택 km", "valid E4", "test E4"], selected_rows),
        "## 3km 혼합 비율",
        "",
        "α는 동네 샘플링을 선택하는 확률입니다. 굵은 α가 도시별 valid E4 선택값입니다.",
        "",
        *table(["도시", "α", "valid E4", "E1", "E3", "E4", "E4 HR@10", "E4 재방문", "E4 신규", "반경 내 정답만 E4"], alpha_rows),
        "## 전역 시간 분할 재검증 (3km)",
        "",
        *table(["도시", "분할", "test 요청", "미등록 정답", "정답 포함률", "재방문율", "E1 1위", "E3 1위", "E4 1위", "E1/E4 순위 τ"], time_rows),
        *table(["도시", "분할", "모델", "E1", "E3", "E4", "E4 HR@10", "재방문", "신규", "등록 신규", "반경 내 정답만"], model_rows),
        "분할마다 이용자·정답·카탈로그가 달라집니다. 순위와 샘플러 차이는 각 분할 안에서 해석합니다.",
        "",
        "## 샘플러 간 E4 차이",
        "",
        "쿼리별로 시드 평균을 구한 뒤 같은 이용자 쿼리를 2,000회 재표집했습니다. "
        "구간은 다중 비교 보정을 적용하기 전의 95% 신뢰구간이며 시드 변동은 위 표에 별도로 표시했습니다.",
        "",
        *table(["도시", "분할", "km", "비교", "NDCG 차이", "95% CI", "상대 차이"], effect_rows),
        "전체 모델·구간별 지표는 [실험 매트릭스](matrix.csv), 모든 paired 비교는 [신뢰구간](paired_bootstrap.csv), "
        "분할별 표본은 [코호트](cohorts.csv), 실행 출처는 [입력 목록](inputs.json)에 저장했습니다.",
        "",
    ])


def plot(scores: pd.DataFrame, path: Path) -> None:
    plt = figures.plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.1), layout="constrained")
    for i, city in enumerate(CITIES):
        data = scores[(scores["split"] == "leave-one-out") & (scores["city"] == city)]
        ax = axes[0, i]
        for model, color in zip(("PersonalFreq", "S0", "S1", "S2"), figures.SERIES):
            rows = data[data["model"] == model].sort_values("radius_km")
            ax.errorbar(rows["radius_km"], rows["e4_ndcg"], yerr=rows["e4_sd"].fillna(0), label=model,
                        marker="o", capsize=3, color=color)
        ax.set(title=f"{city} | radius sensitivity", xlabel="Radius (km)", ylabel="Test E4 NDCG@10", xticks=[2, 3, 5])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax = axes[1, i]
        alpha, models = zip(*ALPHAS)
        rows = data[data["radius_km"] == 3].set_index("model").loc[list(models)]
        ax.errorbar(alpha, rows["e4_ndcg"], yerr=rows["e4_sd"], label="Test", marker="o", capsize=3, color=figures.SERIES[0])
        ax.plot(alpha, rows["valid_e4"], label="Validation", marker="s", linestyle="--", color=figures.SERIES[1])
        ax.set(title=f"{city} | sampler mix at 3 km", xlabel="Geo sampling probability", ylabel="E4 NDCG@10", xticks=alpha)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="완료한 반경·혼합 비율·전역 시간 실험 집계")
    parser.add_argument("--loo-config", default="configs/data.yaml")
    parser.add_argument("--global-config", default="configs/data_global.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--sasrec-config", default="configs/sasrec.yaml")
    args = parser.parse_args()
    ecfg, scfg = load_config(args.eval_config), load_config(args.sasrec_config)
    rows, cohorts, effects, inputs = [], [], [], []
    for path, radii in ((args.loo_config, (2, 3, 5)), (args.global_config, (3,))):
        data = load_config(path)
        for city in CITIES:
            for radius in radii:
                query, cohort, valid, sources = load_group(data, ecfg, scfg, city, radius)
                metrics, diffs = summarize(query, cohort, valid)
                rows.extend(metrics)
                effects.extend(diffs)
                inputs.extend(sources)
                cohorts.append(cohort)
    scores = pd.DataFrame(rows)
    groups = pd.DataFrame(cohorts).set_index(["city", "split", "radius_km"])
    diffs = pd.DataFrame(effects)
    out = Path(ecfg["results_dir"]) / "reports" / "sensitivity" / attempt_id("stage3")
    with publish_directory(out) as staging:
        scores.to_csv(staging / "matrix.csv", index=False)
        groups.to_csv(staging / "cohorts.csv")
        diffs.to_csv(staging / "paired_bootstrap.csv", index=False)
        write_json(staging / "inputs.json", {"runs": inputs})
        plot(scores, staging / "sensitivity.png")
        (staging / "report.md").write_text(render(scores, groups, diffs, inputs), encoding="utf-8")
    print(f"[sensitivity] {len(inputs)}개 실행 → {out / 'report.md'}")


if __name__ == "__main__":
    main()
