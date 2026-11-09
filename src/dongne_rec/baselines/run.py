"""1일차: 베이스라인 4종(MostPop, GeoPop, PersonalFreq, ItemKNN)을 E1~E4로 평가해요.

ItemKNN은 valid의 주 지표로 튜닝하고, 모든 모델은 test로 리포트해요.
결과: results/runs/{city}/R{r}/{run_id}/
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from dongne_rec.baselines.itemknn import ItemKNN
from dongne_rec.baselines.pop import GeoPop, MostPop, PersonalFreq
from dongne_rec.config import load_config, resolve_radius
from dongne_rec.eval import report
from dongne_rec.eval.protocols import PROTOCOLS, evaluate
from dongne_rec.experiments import load_evaluation, new_run
from dongne_rec.storage import publish_directory, write_json


def tune_itemknn(knn: ItemKNN, grid_cfg: dict, run_valid, metric: str, primary: str) -> tuple[ItemKNN, pd.DataFrame]:
    rows = []
    for last_n, decay, include_self in itertools.product(grid_cfg["last_n"], grid_cfg["decay"], grid_cfg["include_self"]):
        if last_n == 1 and decay != 1.0:  # 장소 1개면 decay가 의미 없음
            continue
        res = run_valid(knn.configure(last_n=last_n, decay=decay, include_self=include_self))
        rows.append({"last_n": last_n or "all", "decay": decay, "include_self": include_self,
                     **res.groupby("protocol")[metric].mean().to_dict()})
    grid = pd.DataFrame(rows).sort_values(primary, ascending=False, ignore_index=True)
    best = grid.iloc[0]
    last_n = None if best["last_n"] == "all" else int(best["last_n"])
    return knn.configure(last_n=last_n, decay=float(best["decay"]), include_self=bool(best["include_self"])), grid


def render(city: str, radius: float, ecfg: dict, per_query: pd.DataFrame, cands: dict, grid: pd.DataFrame, n_users: int) -> str:
    metric = ecfg["primary"]["metric"]
    k = ecfg["ks"][0]
    ndcg, hr = report.pivot(per_query, f"ndcg@{k}"), report.pivot(per_query, f"hr@{k}")
    coverage = per_query.loc[per_query["protocol"] == "E4"].groupby("query")["in_candidates"].first().mean()
    fill = cands["test"]["E2_fill"]
    parts = report.breakdown_tables(per_query, metric)
    repeat_share = per_query.loc[per_query["protocol"] == "E4"].groupby("query")["is_repeat"].first().mean()
    gap_share = per_query.loc[per_query["protocol"] == "E4"].groupby("query")["gap_bucket"].first().value_counts(normalize=True)
    inflation = ndcg[["E1"]].div(ndcg["E2"], axis=0).rename(columns={"E1": "E1 ÷ E2"})
    inflation["E3 ÷ E4"] = ndcg["E3"] / ndcg["E4"]
    near = per_query[per_query["query"].isin(per_query.loc[(per_query["protocol"] == "E4") & per_query["in_candidates"], "query"])]
    ndcg_near = report.pivot(near, f"ndcg@{k}").reindex(index=ndcg.index, columns=PROTOCOLS)
    inflation_near = (ndcg_near["E1"] / ndcg_near["E2"]).rename("E1 ÷ E2").to_frame()
    grid_view = grid.set_index(grid.apply(lambda r: f"L={r['last_n']}, decay={r['decay']:g}, self={r['include_self']}", axis=1))

    return "\n".join([
        f"# 베이스라인 — {city}, R={radius:g}km",
        "",
        f"`make baselines`로 생성된 파일이에요. test 유저 {n_users:,}명, E1/E2 negative {ecfg['n_sampled_negatives']}개 "
        f"(seed {ecfg['seed']}), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.",
        "",
        f"- coverage@{radius:g}km (test 정답이 동네 안에 있는 비율 = E4 HR@{k} 상한): **{coverage:.1%}**",
        f"- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: {(fill > 0).mean():.1%} (채운 개수 평균 {fill[fill > 0].mean() if (fill > 0).any() else 0:.1f})",
        f"- 재방문 정답 비율: {repeat_share:.1%}",
        "",
        f"## NDCG@{k} (test)",
        "",
        report.md_table(ndcg[list(PROTOCOLS)], header=report.PROTOCOL_HEADER),
        "",
        f"## HR@{k} (test)",
        "",
        report.md_table(hr[list(PROTOCOLS)], header=report.PROTOCOL_HEADER),
        "",
        "## 프로토콜별 모델 순위 (NDCG)",
        "",
        *report.protocol_rankings(ndcg[list(PROTOCOLS)]),
        "",
        "## 부풀림 배수 (NDCG)",
        "",
        "E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).",
        "",
        report.md_table(inflation, fmt="{:.2f}"),
        "",
        f"### 정답이 동네 안인 쿼리만 ({coverage:.1%}) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기",
        "",
        report.md_table(ndcg_near.join(inflation_near), header={**report.PROTOCOL_HEADER, "E1 ÷ E2": "E1 ÷ E2"}),
        "",
        f"## 재방문 / 신규 정답별 NDCG@{k}",
        "",
        f"재방문 {repeat_share:.1%} / 신규 {1 - repeat_share:.1%}",
        "",
        report.md_table(parts["repeat"]),
        "",
        f"## 직전 체크인과의 시간 간격별 NDCG@{k} (E4)",
        "",
        " / ".join(f"{b} {s:.1%}" for b, s in gap_share.reindex(parts["gap"].columns).items()),
        "",
        report.md_table(parts["gap"]),
        "",
        f"## ItemKNN 튜닝 (valid {metric}, 기준 {ecfg['primary']['protocol']})",
        "",
        report.md_table(grid_view[list(PROTOCOLS)], index_name="설정", header=report.PROTOCOL_HEADER),
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="베이스라인 평가")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--tag", help="비교에 사용할 실행 그룹 이름")
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    ecfg = load_config(args.eval_config)
    radius = resolve_radius(cfg, args.radius_km)
    ks, metric, primary = tuple(ecfg["ks"]), ecfg["primary"]["metric"], ecfg["primary"]["protocol"]

    split, geo, cands, evaluation = load_evaluation(cfg, ecfg, radius)
    out, metadata = new_run("baselines", cfg, ecfg, evaluation, tag=args.tag or None, device="cpu")

    def run(model, stage: str) -> pd.DataFrame:
        return evaluate(model, split.queries(stage), geo, cands[stage], ks, ecfg["batch_size"])

    knn = ItemKNN(topk=ecfg["itemknn"]["topk_neighbors"]).fit(split.train, split.n_items)
    knn, grid = tune_itemknn(knn, ecfg["itemknn"]["grid"], lambda m: run(m, "valid"), metric, primary)

    models = [MostPop(), GeoPop(geo), PersonalFreq()]
    models = [m.fit(split.train, split.n_items) for m in models] + [knn]
    per_query = pd.concat([run(m, "test").assign(model=m.name) for m in models], ignore_index=True)
    per_query = report.with_meta(per_query, split.test)

    summary = {m: report.pivot(per_query, m)[list(PROTOCOLS)].to_dict(orient="index") for m in (*(f"{x}@{k}" for x in ("ndcg", "hr") for k in ks), "mrr")}
    with publish_directory(out) as staging:
        per_query.to_parquet(staging / "per_query.parquet", index=False)
        grid.to_csv(staging / "itemknn_grid.csv", index=False)
        write_json(staging / "summary.json", {"city": cfg["city"], "radius_km": radius, "run_id": metadata["run_id"], **summary})
        write_json(staging / "metadata.json", metadata)
        (staging / "summary.md").write_text(
            render(cfg["city"], radius, ecfg, per_query, cands, grid, len(split.test)), encoding="utf-8",
        )

    print(f"[baselines] {cfg['city']} R={radius:g}km test {metric}")
    print(report.pivot(per_query, metric)[list(PROTOCOLS)].round(4).to_string())
    print(f"[baselines] → {out / 'summary.md'}")


if __name__ == "__main__":
    main()
