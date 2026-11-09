"""베이스라인 + SASRec 실행 결과를 모아 README 표 2 형태로 정리해요.

SASRec은 시드별 평균을 다시 평균 ± 표준편차로 보여줘요.
결과: results/reports/{city}/R{r}/{report_id}/table2.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from html import escape
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from dongne_rec.config import load_config, resolve_radius
from dongne_rec.eval import figures, report
from dongne_rec.eval.protocols import PROTOCOLS
from dongne_rec.eval.stats import paired_bootstrap
from dongne_rec.experiments import attempt_id, saved_runs
from dongne_rec.storage import publish_directory, write_json
from dongne_rec.train.samplers import SAMPLER_IDS

S0, S1, S2 = "SASRec-S0 (random)", "SASRec-S1 (geo)", "SASRec-S2 (mixed)"
FOCUS = ["PersonalFreq", S0, S1, S2]
SHORT = {S0: "S0 random", S1: "S1 geo", S2: "S2 mixed"}
SLICES = {  # 이름: (프로토콜, 재방문 여부 필터)
    "E1": ("E1", None), "E2": ("E2", None), "E3": ("E3", None), "E4": ("E4", None),
    "E4 new": ("E4", False), "E4 repeat": ("E4", True),
}


def select_runs(
    results_dir: Path, city: str, radius: float, explicit: list[Path] | None = None,
) -> list[tuple[Path, dict | None]]:
    sampler_order = {s: i for i, s in enumerate(SAMPLER_IDS)}
    if explicit:
        selected = []
        for path in dict.fromkeys(p.resolve() for p in explicit):
            manifest = path / "metadata.json"
            info = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
            evaluation = info["evaluation"] if info else json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if evaluation["city"] != city or evaluation["radius_km"] != radius:
                raise ValueError(f"도시 또는 반경이 다른 실행입니다. {path}")
            selected.append((path, info))
    else:
        latest = {info["experiment_id"]: (path, info) for path, info in saved_runs(results_dir, city, radius) if info["tag"] is None}
        selected = sorted(latest.values(), key=lambda pair: (
            sampler_order.get(pair[1]["sampler"], -1), pair[1]["seed"] if pair[1]["seed"] is not None else -1, pair[0].name,
        ))
        if not selected:
            baseline = results_dir / "baselines" / city / f"R{radius:g}"
            runs = (results_dir / "sasrec" / city / f"R{radius:g}").glob("*/per_query.parquet")
            paths = sorted((p.parent for p in runs), key=lambda p: (sampler_order.get(p.name.rsplit("_s", 1)[0], 99), p.name))
            if (baseline / "per_query.parquet").is_file():
                paths.insert(0, baseline)
            selected = [(path, None) for path in paths]
    if not selected:
        raise FileNotFoundError(f"{results_dir}에 {city} R={radius:g}km 결과가 없어요 (make baselines / make train 먼저).")
    evaluation_ids = {info["evaluation"]["evaluation_id"] if info else None for _, info in selected}
    if len(evaluation_ids) != 1:
        raise ValueError("평가 조건이 다른 실행이 포함됐습니다. 같은 데이터·분할·후보 설정의 실행을 --run으로 선택하세요.")
    experiments = [info["experiment_id"] for _, info in selected if info]
    if len(experiments) != len(set(experiments)):
        raise ValueError("같은 설정과 시드의 재실행이 포함됐습니다. 실험당 실행 하나를 선택하세요.")
    return selected


def load_results(runs: list[tuple[Path, dict | None]]) -> pd.DataFrame:
    frames = []
    variants = defaultdict(set)
    for path, info in runs:
        frame = pd.read_parquet(path / "per_query.parquet")
        if "seed" not in frame:
            frame = frame.assign(seed=-1)
        variant = info["variant_id"] if info else None
        details = info or json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if info and info["sampler"] == "mixed":
            ratio = info["configs"]["sasrec"]["train"]["geo_ratio"]
            if ratio != 0.5:
                frame["model"] = frame["model"] + f" α={ratio:g}"
        if details.get("tag"):
            tag = escape(" ".join(str(details["tag"]).split())).replace("|", "&#124;")
            frame["model"] = frame["model"] + f" [{tag}]"
        for model in frame["model"].unique():
            variants[model].add(variant)
        frames.append((frame, variant))

    results = []
    for frame, variant in frames:
        if variant:
            frame["model"] = frame["model"].map(lambda name: f"{name} [{variant[:8]}]" if len(variants[name]) > 1 else name)
        results.append(frame)
    per_query = pd.concat(results, ignore_index=True)
    reference = None
    for _, group in per_query.groupby(["model", "seed"], sort=False):
        keys = group.set_index(["query", "protocol"])[["user_idx", "is_repeat", "gap_bucket", "in_candidates"]].sort_index()
        if keys.index.has_duplicates or (reference is not None and not reference.equals(keys)):
            raise ValueError("비교에 사용할 쿼리나 후보 범위가 일치하지 않습니다.")
        reference = keys
    return per_query


def mean_std_table(per_query: pd.DataFrame, metric: str, columns: str = "protocol") -> pd.DataFrame:
    """모델 × columns, 시드가 2개 이상이면 'mean ± std' 문자열."""
    order = list(dict.fromkeys(per_query["model"]))
    if not order:
        return pd.DataFrame()
    per_seed = per_query.groupby(["model", "seed", columns], observed=True)[metric].mean()
    agg = per_seed.groupby(["model", columns], observed=True).agg(["mean", "std", "count"])

    def fmt(r: pd.Series) -> str:
        return f"{r['mean']:.4f}" if r["count"] < 2 else f"{r['mean']:.4f} ± {r['std']:.4f}"

    return agg.apply(fmt, axis=1).unstack(columns).reindex(order)


def seed_counts(per_query: pd.DataFrame) -> pd.Series:
    return per_query[per_query["seed"] >= 0].groupby("model")["seed"].nunique()


def query_matrix(per_query: pd.DataFrame, metric: str, protocol: str, repeat: bool | None = None) -> pd.DataFrame:
    """쿼리 × 모델 지표 (SASRec은 시드 평균)."""
    d = per_query[per_query["protocol"] == protocol]
    if repeat is not None:
        d = d[d["is_repeat"] == repeat]
    return d.groupby(["query", "model"])[metric].mean().unstack("model")


def bootstrap_table(per_query: pd.DataFrame, metric: str, pairs: list[tuple[str, str]], slices: list[str]) -> pd.DataFrame:
    """(base, other) 쌍 × 지표 구간별 paired bootstrap."""
    rows = []
    for name in slices:
        protocol, repeat = SLICES[name]
        m = query_matrix(per_query, metric, protocol, repeat)
        for base, other in pairs:
            if base in m and other in m:
                rows.append({"comparison": f"{SHORT.get(other, other)} vs {SHORT.get(base, base)}", "slice": name,
                             **paired_bootstrap(m[base].to_numpy(), m[other].to_numpy())})
    return pd.DataFrame(rows)


def bootstrap_md(table: pd.DataFrame) -> str:
    lines = ["| 비교 | 구간 | 차이 (상대) | 95% CI | p |", "|---|---|---|---|---|"]
    for r in table.itertuples():
        mark = " ✱" if r.p < 0.05 else ""
        lines.append(f"| {r.comparison} | {r.slice} | {r.diff:+.4f} ({r.rel:+.1%}){mark} | [{r.lo:+.4f}, {r.hi:+.4f}] | {r.p:.3f} |")
    return "\n".join(lines)


def render(city: str, radius: float, per_query: pd.DataFrame, k: int, out: Path) -> str:
    near_q = per_query.loc[(per_query["protocol"] == "E4") & per_query["in_candidates"], ["model", "seed", "query"]]
    near = per_query.merge(near_q, on=["model", "seed", "query"])
    full = per_query[per_query["protocol"].isin(["E3", "E4"])].assign(
        group=lambda d: d["protocol"] + " " + d["is_repeat"].map({True: "재방문", False: "신규"})
    )
    seeds = seed_counts(per_query)
    metric = f"ndcg@{k}"
    ndcg = report.pivot(per_query, metric)[list(PROTOCOLS)]
    tau, _ = kendalltau(ndcg["E1"], ndcg["E4"])
    labels = {m: SHORT.get(m, m.split("(")[0] if m.startswith("ItemKNN(") else m) for m in ndcg.index}
    figures.protocol_ranks(ndcg, FOCUS, f"{city}: model rank by evaluation protocol (R={radius:g}km)", out / "fig_protocol_ranks.png", labels)

    h1 = bootstrap_table(per_query, metric, [(S0, S1), (S0, S2)], list(SLICES))
    vs_pf = bootstrap_table(per_query, metric, [("PersonalFreq", m) for m in (S0, S1, S2)], ["E4", "E4 new", "E4 repeat"])
    h1_section = []
    if len(h1):
        colors = {c: figures.SERIES[2 + i] for i, c in enumerate(dict.fromkeys(h1["comparison"]))}  # S1 = 3번, S2 = 4번 슬롯
        figures.sampler_effects(h1, colors, f"{city}: effect of geo negatives (H1)", out / "fig_sampler_effects.png")
        h1_section = [
            f"## H1: 샘플러 효과 — S0 대비 NDCG@{k} 차이 (쿼리 단위 paired bootstrap 2,000회, SASRec은 쿼리별 시드 평균)",
            "",
            "![sampler effects](fig_sampler_effects.png)",
            "",
            bootstrap_md(h1),
            "",
            "✱ p < 0.05. 신뢰구간은 쿼리(유저) 표본 변동만 반영하고 시드 변동은 위 표의 ± 표준편차로 따로 봐요.",
            "",
        ]

    return "\n".join([
        f"# 모델 비교 — {city}, R={radius:g}km (test)",
        "",
        "`make compare`로 생성된 파일이에요. 베이스라인은 결정적, SASRec은 시드별 평균의 평균 ± 표준편차.",
        "선택한 실행은 [입력 목록](inputs.json)에 기록했습니다.",
        "",
        *(f"- {m}: 시드 {n}개" for m, n in seeds.items()),
        "",
        f"## 프로토콜별 순위 (H2)",
        "",
        "![protocol ranks](fig_protocol_ranks.png)",
        "",
        *report.protocol_rankings(ndcg),
        "",
        f"E1 순위와 E4 순위의 Kendall τ = {tau:.2f} (1이면 같은 순서, 0이면 무관)",
        "",
        *h1_section,
        f"## PersonalFreq 대비 SASRec (NDCG@{k}, paired bootstrap)",
        "",
        bootstrap_md(vs_pf),
        "",
        f"## 표 2. NDCG@{k}",
        "",
        report.md_table(mean_std_table(per_query, f"ndcg@{k}")[list(PROTOCOLS)], fmt="{}", header=report.PROTOCOL_HEADER),
        "",
        f"## HR@{k}",
        "",
        report.md_table(mean_std_table(per_query, f"hr@{k}")[list(PROTOCOLS)], fmt="{}", header=report.PROTOCOL_HEADER),
        "",
        f"## 표 3. 재방문 / 신규 정답별 NDCG@{k}",
        "",
        report.md_table(mean_std_table(full, f"ndcg@{k}", columns="group").reindex(columns=["E3 재방문", "E3 신규", "E4 재방문", "E4 신규"]), fmt="{}"),
        "",
        f"## 정답이 동네 안인 쿼리만 NDCG@{k} — negative 난이도 효과",
        "",
        report.md_table(mean_std_table(near, f"ndcg@{k}").reindex(index=ndcg.index, columns=PROTOCOLS), fmt="{}", header=report.PROTOCOL_HEADER),
        "",
        f"## 직전 체크인과의 시간 간격별 NDCG@{k} (E4)",
        "",
        report.md_table(mean_std_table(per_query[per_query["protocol"] == "E4"], f"ndcg@{k}", columns="gap_bucket"), fmt="{}"),
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 비교 표")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--run", action="append", type=Path, help="비교할 실행 디렉터리. 실행마다 반복해 지정합니다.")
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    ecfg = load_config(args.eval_config)
    radius = resolve_radius(cfg, args.radius_km)
    results_dir = Path(ecfg["results_dir"])
    try:
        runs = select_runs(results_dir, cfg["city"], radius, args.run)
        per_query = load_results(runs)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    first = runs[0][1]
    k = ecfg["ks"][0]
    metric = ecfg["primary"]["metric"]
    required = [f"ndcg@{k}", f"hr@{k}", metric]
    if any(column not in per_query or per_query[column].isna().any() for column in required):
        parser.error("선택한 실행에 필요한 지표가 없습니다. --eval-config의 ks와 primary를 확인하세요.")
    out = results_dir / "reports" / cfg["city"] / f"R{radius:g}" / attempt_id("compare")
    with publish_directory(out) as staging:
        write_json(staging / "inputs.json", {
            "city": cfg["city"], "radius_km": radius, "k": k,
            "evaluation_id": first["evaluation"]["evaluation_id"] if first else None,
            "runs": [
                {"path": str(path), "run_id": info["run_id"] if info else None}
                for path, info in runs
            ],
        })
        (staging / "table2.md").write_text(render(cfg["city"], radius, per_query, k, staging), encoding="utf-8")
    print(mean_std_table(per_query, metric)[list(PROTOCOLS)].to_string())
    print(f"[compare] → {out / 'table2.md'}")


if __name__ == "__main__":
    main()
