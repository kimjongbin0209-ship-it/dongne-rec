"""베이스라인 + SASRec 실행 결과를 모아 README 표 2 형태로 정리해요.

SASRec은 시드별 평균을 다시 평균 ± 표준편차로 보여줘요.
결과: results/summary/{city}/R{r}/table2.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from dongne_rec.config import load_config
from dongne_rec.eval import figures, report
from dongne_rec.eval.protocols import PROTOCOLS
from dongne_rec.eval.stats import paired_bootstrap
from dongne_rec.train.samplers import SAMPLER_IDS

S0, S1, S2 = "SASRec-S0 (random)", "SASRec-S1 (geo)", "SASRec-S2 (mixed)"
FOCUS = ["PersonalFreq", S0, S1, S2]
SHORT = {S0: "S0 random", S1: "S1 geo", S2: "S2 mixed"}
SLICES = {  # 이름: (프로토콜, 재방문 여부 필터)
    "E1": ("E1", None), "E2": ("E2", None), "E3": ("E3", None), "E4": ("E4", None),
    "E4 new": ("E4", False), "E4 repeat": ("E4", True),
}


def load_results(results_dir: Path, city: str, radius: float) -> pd.DataFrame:
    frames = []
    baseline = results_dir / "baselines" / city / f"R{radius:g}" / "per_query.parquet"
    if baseline.exists():
        frames.append(pd.read_parquet(baseline).assign(seed=-1))
    runs = (results_dir / "sasrec" / city / f"R{radius:g}").glob("*/per_query.parquet")
    sampler_order = {s: i for i, s in enumerate(SAMPLER_IDS)}  # S0, S1, S2 순
    frames += [pd.read_parquet(p) for p in sorted(runs, key=lambda p: (sampler_order.get(p.parent.name.rsplit("_s", 1)[0], 99), p.parent.name))]
    if not frames:
        raise FileNotFoundError(f"{results_dir}에 {city} R={radius:g}km 결과가 없어요 (make baselines / make train 먼저).")
    return pd.concat(frames, ignore_index=True)


def mean_std_table(per_query: pd.DataFrame, metric: str, columns: str = "protocol") -> pd.DataFrame:
    """모델 × columns, 시드가 2개 이상이면 'mean ± std' 문자열."""
    order = list(dict.fromkeys(per_query["model"]))
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
    labels = {m: SHORT.get(m, m.split("(")[0]) for m in ndcg.index}
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
        report.md_table(mean_std_table(full, f"ndcg@{k}", columns="group")[["E3 재방문", "E3 신규", "E4 재방문", "E4 신규"]], fmt="{}"),
        "",
        f"## 정답이 동네 안인 쿼리만 NDCG@{k} — negative 난이도 효과",
        "",
        report.md_table(mean_std_table(near, f"ndcg@{k}")[list(PROTOCOLS)], fmt="{}", header=report.PROTOCOL_HEADER),
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
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    ecfg = load_config(args.eval_config)
    radius = args.radius_km if args.radius_km is not None else cfg["geo"]["radius_km"]
    results_dir = Path(ecfg["results_dir"])
    per_query = load_results(results_dir, cfg["city"], radius)

    out = results_dir / "summary" / cfg["city"] / f"R{radius:g}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table2.md").write_text(render(cfg["city"], radius, per_query, ecfg["ks"][0], out), encoding="utf-8")
    metric = ecfg["primary"]["metric"]
    print(mean_std_table(per_query, metric)[list(PROTOCOLS)].to_string())
    print(f"[compare] → {out / 'table2.md'}")


if __name__ == "__main__":
    main()
