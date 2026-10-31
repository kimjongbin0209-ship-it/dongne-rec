"""4일차: 서빙 latency·정확도 벤치 (batch=1).

python -m dongne_rec.serving.bench --city NYC
요청 = test 쿼리 전체(최근 시퀀스 + 직전 장소 좌표). 정답이 있으니 정확도도 같이 재요.
  표 4: 인코더 백엔드 × 스레드 (2단계 구성)
  표 5: 파이프라인 구성 (2단계 FAISS / numpy, 1-b 생략, 지리 필터 없음)
구성마다 새 프로세스에서 재요 (torch·FAISS OpenMP 충돌 방지, 측정 간 간섭 제거).
결과: results/serving/{city}/R{r}/{sampler}_s{seed}/ bench.md, bench.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dongne_rec.config import load_config
from dongne_rec.data.split import Queries, load_split
from dongne_rec.serving.pipeline import TwoStageRecommender, artifacts_dir, load_recommender

STAGES = ("geo_filter", "narrow", "rerank", "total")
SCALING_SIZES = (1_000, 10_000, 100_000, 1_000_000)


def scoring_cost(hidden: int, sizes=SCALING_SIZES, k: int = 10, repeats: int = 30, seed: int = 0) -> list[dict]:
    """후보 n개의 점수 계산(e · h)과 상위 k 선택 비용 — 카탈로그가 커지면 Stage 2가 얼마나 느려지는지.

    인코더 비용은 후보 수와 무관하니 이 부분만 따로 재요. FAISS IVF(nprobe=8)로 같은 n에서 상위 200개를 뽑는 비용도 같이 봐요.
    """
    import time

    import faiss

    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        emb = rng.standard_normal((n, hidden), dtype=np.float32)
        h = rng.standard_normal(hidden, dtype=np.float32)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            scores = emb @ h
            np.argpartition(-scores, k)[:k]
            times.append((time.perf_counter_ns() - t0) / 1e6)
        nlist = max(1, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatIP(hidden)
        ivf = faiss.IndexIVFFlat(quantizer, hidden, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(emb[: min(n, 50 * nlist)])
        ivf.add(emb)
        ivf.nprobe = 8
        ivf_times = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            ivf.search(h[None], 200)
            ivf_times.append((time.perf_counter_ns() - t0) / 1e6)
        rows.append({"n": n, "brute_p50": float(np.median(times)), "ivf_top200_p50": float(np.median(ivf_times))})
    return rows


def run_requests(rec: TwoStageRecommender, q: Queries, warmup: int, k: int = 10) -> pd.DataFrame:
    def call(i: int):
        last = q.last_items[i]
        return rec.recommend(q.histories[i], rec.lat[last], rec.lon[last], k=k)

    for i in range(min(warmup, len(q))):
        call(i)
    rows = []
    for i in range(len(q)):
        r = call(i)
        hit = np.flatnonzero(r.items == q.targets[i])
        rank = hit[0] + 1 if len(hit) else 0
        rows.append({**r.timings_ms, "n_geo": r.n_geo, "n_candidates": r.n_candidates, "radius_km": r.radius_km,
                     "in_candidates": bool(np.isin(q.targets[i], r.candidates)),
                     f"hr@{k}": float(rank > 0), f"ndcg@{k}": 1 / np.log2(rank + 1) if rank else 0.0})
    return pd.DataFrame(rows)


def requests_job(cfg: dict, art: Path, backend: str, threads: int, pipeline_kwargs: dict, warmup: int,
                 radius_override: float | None = None) -> dict:
    rec, _, _ = load_recommender(art, backend, threads, **pipeline_kwargs)
    if radius_override:
        rec.radius_km = radius_override
    return summarize(run_requests(rec, load_split(cfg).test, warmup))


def isolated(fn, *args):
    """새 프로세스에서 fn(*args)를 실행해 결과를 돌려줘요."""
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def summarize(df: pd.DataFrame, k: int = 10) -> dict:
    out = {f"{s}_{p}": float(np.percentile(df[s], int(p[1:]))) for s in STAGES for p in ("p50", "p95", "p99")}
    out.update({
        "total_mean": float(df["total"].mean()),
        f"ndcg@{k}": float(df[f"ndcg@{k}"].mean()), f"hr@{k}": float(df[f"hr@{k}"].mean()),
        "stage1_recall": float(df["in_candidates"].mean()),
        "n_candidates_p50": float(df["n_candidates"].median()), "n_geo_p50": float(df["n_geo"].median()),
        "narrowed_share": float((df["n_geo"] > df["n_candidates"]).mean()), "expanded_share": float((df["radius_km"] > df["radius_km"].min()).mean()),
    })
    return out


def ms(x: float) -> str:
    return f"{x:.2f}"


def render(meta: dict, backend_rows: list[dict], pipeline_rows: list[dict], sweep_rows: list[dict], scaling: list[dict],
           offline: dict, n: int, pcfg: dict) -> str:
    ref = next(r for r in pipeline_rows if r["variant"].startswith("2단계 (FAISS)"))
    return "\n".join([
        f"# 서빙 벤치 — {meta['city']}, R={meta['radius_km']:g}km, {meta['model']} seed {meta['seed']}",
        "",
        f"`make bench`로 생성된 파일이에요. 프로세스 내 측정, batch=1, test 요청 {n:,}건 (warmup 별도), {platform.platform()}.",
        f"후보 N={pcfg['n_candidates']}, Stage 1-b 쿼리 = 최근 {pcfg['query_last_n']}개 장소 임베딩 평균.",
        "",
        "## 표 4. 인코더 백엔드 (2단계 구성)",
        "",
        "| 인코더 | 스레드 | 1-b 검색 | 인코더 크기 | Stage 2 p50 / p99 (ms) | E2E p50 / p99 (ms) | NDCG@10 | HR@10 |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r['backend']} | {r['threads']} | {r['retrieval']} | {r['size_mb']:.1f}MB | {ms(r['rerank_p50'])} / {ms(r['rerank_p99'])} | "
          f"{ms(r['total_p50'])} / {ms(r['total_p99'])} | {r['ndcg@10']:.4f} | {r['hr@10']:.4f} |" for r in backend_rows],
        "",
        "torch 인코더는 FAISS와 한 프로세스에서 OpenMP 런타임이 충돌해서(macOS) numpy 검색으로 재요. 두 검색의 후보 집합은 같아요.",
        "",
        f"아이템 임베딩 {meta['size_mb']['item_emb.npy']:.1f}MB(fp32)는 모든 구성에서 후보 점수 계산에 공통으로 써요. "
        f"ONNX 변환 오차 max|onnx − torch|: fp32 {meta['max_abs_diff']['fp32']:.1e}, int8 {meta['max_abs_diff']['int8']:.1e}.",
        "",
        "## 표 5. 파이프라인 구성 (인코더 onnx-fp32, 단일 스레드)",
        "",
        "| 구성 | 재랭킹 후보 수 (median) | 정답이 후보에 남은 비율 | 1-a / 1-b / 2 p50 (ms) | E2E p50 / p99 (ms) | NDCG@10 | 오프라인 기준 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r['variant']} | {r['n_candidates_p50']:.0f} | {r['stage1_recall']:.1%} | {ms(r['geo_filter_p50'])} / {ms(r['narrow_p50'])} / {ms(r['rerank_p50'])} | "
          f"{ms(r['total_p50'])} / {ms(r['total_p99'])} | {r['ndcg@10']:.4f} | {r['offline']} |" for r in pipeline_rows],
        "",
        f"- 오프라인 기준: 같은 체크포인트의 test {offline['E4']:.4f} (E4, 동네 전체 랭킹), {offline['E3']:.4f} (E3, 전체 랭킹). "
        f"'1-b 생략'과 E4의 작은 차이는 반경 안 장소가 k개 미만이라 반경을 넓힌 요청({ref['expanded_share']:.1%})에서 와요 "
        "(E4는 반경 밖 정답을 miss로 셈).",
        f"- 2단계 (FAISS): 반경 안 장소 median {ref['n_geo_p50']:.0f}개, 그중 N개 초과로 1-b가 후보를 줄인 요청 {ref['narrowed_share']:.1%}, "
        f"반경을 넓힌 요청 {ref['expanded_share']:.1%}.",
        "",
        "## 후보 수 N에 따른 trade-off (2단계 FAISS, onnx-fp32 단일 스레드)",
        "",
        "| N | 정답이 후보에 남은 비율 | 1-b p50 (ms) | E2E p50 (ms) | NDCG@10 |",
        "|---|---|---|---|---|",
        *[f"| {r['n_candidates']} | {r['stage1_recall']:.1%} | {ms(r['narrow_p50'])} | {ms(r['total_p50'])} | {r['ndcg@10']:.4f} |" for r in sweep_rows],
        "",
        f"## 카탈로그가 커지면 — 후보 n개 점수 계산 비용 (hidden {meta['hidden']}, 합성 임베딩)",
        "",
        "| 후보 수 n | 전체 점수 + 상위 10 (ms) | FAISS IVF 상위 200 (nprobe 8, ms) |",
        "|---|---|---|",
        *[f"| {r['n']:,} | {ms(r['brute_p50'])} | {ms(r['ivf_top200_p50'])} |" for r in scaling],
        "",
        "인코더 비용(표 4의 Stage 2, 약 0.5~0.8ms)은 후보 수와 무관하고, 전체 점수 계산만 n에 비례해 늘어요.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="서빙 벤치")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--city")
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    ecfg, scfg = load_config(args.eval_config), load_config(args.serving_config)
    radius, sampler, seed = cfg["geo"]["radius_km"], scfg["model"]["sampler"], scfg["model"]["seed"]
    art = artifacts_dir(scfg, cfg["city"], radius, sampler, seed)
    n_requests = len(load_split(cfg).test)
    pcfg, bcfg = scfg["pipeline"], scfg["bench"]
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    offline = json.loads((Path(meta["run"]) / "summary.json").read_text(encoding="utf-8"))["test"]["ndcg@10"]
    sizes = {"onnx-fp32": meta["size_mb"]["encoder_fp32.onnx"], "onnx-int8": meta["size_mb"]["encoder_int8.onnx"]}
    torch_size = (Path(meta["run"]) / "model.pt").stat().st_size / 1e6

    backend_rows = []
    for backend in bcfg["backends"]:
        kwargs = {**pcfg, "retrieval": "numpy"} if backend.startswith("torch-") else pcfg
        for threads in ([1] if backend == "torch-mps" else bcfg["threads"]):
            row = {"backend": backend, "threads": "기본" if threads == 0 else threads, "retrieval": kwargs["retrieval"],
                   "size_mb": sizes.get(backend, torch_size), **isolated(requests_job, cfg, art, backend, threads, kwargs, bcfg["warmup"])}
            backend_rows.append(row)
            print(f"[bench] {backend:<10} threads={row['threads']}: E2E p50 {row['total_p50']:.2f}ms p99 {row['total_p99']:.2f}ms, "
                  f"stage2 p50 {row['rerank_p50']:.2f}ms, NDCG@10 {row['ndcg@10']:.4f}")

    variants = [
        ("2단계 (FAISS)", {}, "E4"),
        ("2단계 (numpy brute force)", {"retrieval": "numpy"}, "E4"),
        ("지리 필터 + 반경 안 전체 재랭킹 (1-b 생략)", {"n_candidates": 10**9}, "E4"),
        ("지리 필터 없음, 전체 카탈로그 재랭킹", {"n_candidates": 10**9, "radius_override": 1e5}, "E3"),
    ]
    pipeline_rows = []
    for name, override, ref in variants:
        kwargs = {**pcfg, **{k: v for k, v in override.items() if k != "radius_override"}}
        summary = isolated(requests_job, cfg, art, "onnx-fp32", 1, kwargs, bcfg["warmup"], override.get("radius_override"))
        row = {"variant": name, "offline": f"{ref} {offline[ref]:.4f}", **summary}
        pipeline_rows.append(row)
        print(f"[bench] {name}: candidates {row['n_candidates_p50']:.0f}, recall {row['stage1_recall']:.1%}, "
              f"E2E p50 {row['total_p50']:.2f}ms, NDCG@10 {row['ndcg@10']:.4f} (offline {row['offline']})")

    sweep_rows = []
    for n_cand in (50, 100, 200, 500, 1000):
        summary = isolated(requests_job, cfg, art, "onnx-fp32", 1, {**pcfg, "n_candidates": n_cand}, bcfg["warmup"])
        sweep_rows.append({"n_candidates": n_cand, **{k: v for k, v in summary.items() if k != "n_candidates_p50"}})
    print("[bench] N sweep: " + ", ".join(f"N={r['n_candidates']} recall {r['stage1_recall']:.1%} NDCG {r['ndcg@10']:.4f}" for r in sweep_rows))

    scaling = isolated(scoring_cost, meta["hidden"])
    print("[bench] scoring cost: " + ", ".join(f"n={r['n']:,} {r['brute_p50']:.2f}ms (IVF {r['ivf_top200_p50']:.2f}ms)" for r in scaling))

    out = Path(ecfg["results_dir"]) / "serving" / cfg["city"] / f"R{radius:g}" / f"{sampler}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps({"meta": meta, "backends": backend_rows, "pipelines": pipeline_rows,
                                                "n_sweep": sweep_rows, "scaling": scaling}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "bench.md").write_text(render(meta, backend_rows, pipeline_rows, sweep_rows, scaling, offline, n_requests, pcfg), encoding="utf-8")
    print(f"[bench] → {out / 'bench.md'}")


if __name__ == "__main__":
    main()
