"""베이스라인 + SASRec 실행 결과를 모아 README 표 2 형태로 정리해요.

SASRec은 시드별 평균을 다시 평균 ± 표준편차로 보여줘요.
결과: results/summary/{city}/R{r}/table2.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dongne_rec.config import load_config
from dongne_rec.eval import report
from dongne_rec.eval.protocols import PROTOCOLS


def load_results(results_dir: Path, city: str, radius: float) -> pd.DataFrame:
    frames = []
    baseline = results_dir / "baselines" / city / f"R{radius:g}" / "per_query.parquet"
    if baseline.exists():
        frames.append(pd.read_parquet(baseline).assign(seed=-1))
    frames += [pd.read_parquet(p) for p in sorted((results_dir / "sasrec" / city / f"R{radius:g}").glob("*/per_query.parquet"))]
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


def render(city: str, radius: float, per_query: pd.DataFrame, k: int) -> str:
    near_q = per_query.loc[(per_query["protocol"] == "E4") & per_query["in_candidates"], ["model", "seed", "query"]]
    near = per_query.merge(near_q, on=["model", "seed", "query"])
    full = per_query[per_query["protocol"].isin(["E3", "E4"])].assign(
        group=lambda d: d["protocol"] + " " + d["is_repeat"].map({True: "재방문", False: "신규"})
    )
    seeds = seed_counts(per_query)
    return "\n".join([
        f"# 모델 비교 — {city}, R={radius:g}km (test)",
        "",
        "`make compare`로 생성된 파일이에요. 베이스라인은 결정적, SASRec은 시드별 평균의 평균 ± 표준편차.",
        "",
        *(f"- {m}: 시드 {n}개" for m, n in seeds.items()),
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
    (out / "table2.md").write_text(render(cfg["city"], radius, per_query, ecfg["ks"][0]), encoding="utf-8")
    metric = ecfg["primary"]["metric"]
    print(mean_std_table(per_query, metric)[list(PROTOCOLS)].to_string())
    print(f"[compare] → {out / 'table2.md'}")


if __name__ == "__main__":
    main()
