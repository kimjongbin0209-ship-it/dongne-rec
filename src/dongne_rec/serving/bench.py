"""서빙 지연과 추천 품질을 검증합니다 (batch=1).

python -m dongne_rec.serving.bench --city NYC
API와 같은 파이프라인 설정으로 백엔드를 측정하고, 고정 반경과 확장을 별도로 비교합니다.
각 측정은 별도 프로세스에서 실행합니다. 결과는 실행별 bench.md와 bench.json에 기록합니다.
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

from dongne_rec.config import load_config, resolve_radius
from dongne_rec.data.split import Queries, load_split_inputs, split_fingerprint
from dongne_rec.eval.metrics import expected_metrics
from dongne_rec.experiments import attempt_id, environment_metadata, git_metadata
from dongne_rec.serving.pipeline import (
    DEFAULT_SERVING_CONFIG, TwoStageRecommender, artifacts_dir, load_recommender, pipeline_settings,
)
from dongne_rec.storage import fingerprint, publish_directory, write_json

STAGES = ("geo_filter", "narrow", "rerank", "total")
SCALING_SIZES = (1_000, 10_000, 100_000, 1_000_000)
E4_ATOL = 1e-4


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


def run_requests(
    rec: TwoStageRecommender, q: Queries, warmup: int, k: int = 10, radius_override: float | None = None,
) -> pd.DataFrame:
    radius = rec.radius_km if radius_override is None else radius_override

    def call(i: int):
        last = q.last_items[i]
        return rec.recommend(q.histories[i], rec.lat[last], rec.lon[last], k=k, radius_km=radius)

    for i in range(min(warmup, len(q))):
        call(i)
    rows = []
    greater, ties = np.zeros(len(q), dtype=np.int64), np.zeros(len(q), dtype=np.int64)
    for i in range(len(q)):
        r = call(i)
        target = q.targets[i]
        match = np.flatnonzero(r.candidates == target)
        if match.size:
            target_score = r.candidate_scores[match[0]]
            greater[i] = np.count_nonzero(r.candidate_scores > target_score)
            ties[i] = np.count_nonzero(r.candidate_scores == target_score) - 1
        hit = np.flatnonzero(r.items == target)
        rank = int(hit[0]) + 1 if hit.size else 0
        rows.append({
            **r.timings_ms, "n_geo": r.n_geo, "n_candidates": r.n_candidates, "radius_km": r.radius_km,
            "expanded": r.radius_km > radius, "in_candidates": bool(match.size),
            f"served_hr@{k}": float(rank > 0), f"served_ndcg@{k}": 1 / np.log2(rank + 1) if rank else 0.0,
        })
    df = pd.DataFrame(rows)
    metrics = expected_metrics(greater, ties, (k,))
    for metric in (f"hr@{k}", f"ndcg@{k}"):
        df[metric] = np.where(df["in_candidates"], metrics[metric], 0.0)
    return df


def requests_job(q: Queries, art: Path, backend: str, threads: int, pipeline_kwargs: dict, warmup: int,
                 radius_override: float | None = None) -> dict:
    rec, _, _ = load_recommender(art, backend, threads, **pipeline_kwargs)
    return summarize(run_requests(rec, q, warmup, radius_override=radius_override))


def isolated(fn, *args):
    """새 프로세스에서 fn(*args)를 실행해 결과를 돌려줘요."""
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def summarize(df: pd.DataFrame, k: int = 10) -> dict:
    out = {f"{s}_{p}": float(np.percentile(df[s], int(p[1:]))) for s in STAGES for p in ("p50", "p95", "p99")}
    out.update({
        "total_mean": float(df["total"].mean()),
        **{name: float(df[name].mean()) for name in (f"ndcg@{k}", f"hr@{k}", f"served_ndcg@{k}", f"served_hr@{k}")},
        "stage1_recall": float(df["in_candidates"].mean()),
        "n_candidates_p50": float(df["n_candidates"].median()), "n_geo_p50": float(df["n_geo"].median()),
        "narrowed_share": float((df["n_geo"] > df["n_candidates"]).mean()),
        "expanded_share": float(df["expanded"].mean()),
        "expanded_requests": int(df["expanded"].sum()),
        "max_radius_used_km": float(df["radius_km"].max()),
    })
    return out


def ms(x: float) -> str:
    return f"{x:.2f}"


def render(
    meta: dict, backend_rows: list[dict], pipeline_rows: list[dict], sweep_rows: list[dict], scaling: list[dict],
    n: int, pcfg: dict, parity: dict, skipped: list[str],
) -> str:
    lines = [
        f"# 서빙 벤치 — {meta['city']}, R={meta['radius_km']:g}km, {meta['model']} seed {meta['seed']}",
        "",
        f"프로세스 내에서 batch 1, test 요청 {n:,}건을 측정했습니다. 실행 환경은 {platform.platform()}입니다.",
        f"API 공통 설정은 `retrieval={pcfg['retrieval']}`, `radius_mode={pcfg['radius_mode']}`입니다.",
        "NDCG@10과 HR@10은 오프라인과 동일하게 동점 순위의 기댓값으로 계산합니다. "
        "실제 반환 순위로 계산한 지표는 bench.json의 `served_*` 필드에 기록합니다.",
        "",
        "## 인코더 백엔드 (API 공통 정책)",
        "",
        "| 인코더 | 스레드 | 후보 방식 | 인코더 크기 | 재랭킹 p50 / p99 (ms) | 전체 p50 / p99 (ms) | NDCG@10 | HR@10 |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r['backend']} | {r['threads']} | {r['retrieval']} | {r['size_mb']:.1f}MB | {ms(r['rerank_p50'])} / {ms(r['rerank_p99'])} | "
          f"{ms(r['total_p50'])} / {ms(r['total_p99'])} | {r['ndcg@10']:.4f} | {r['hr@10']:.4f} |" for r in backend_rows],
        "",
        *([f"이 환경과 정책에서 측정을 생략한 백엔드는 {', '.join(skipped)}입니다.", ""] if skipped else []),
        f"인코더 파일 크기는 ONNX fp32 {meta['size_mb']['encoder_fp32.onnx']:.1f}MB, "
        f"int8 {meta['size_mb']['encoder_int8.onnx']:.1f}MB입니다. 후보 점수는 공통 fp32 아이템 임베딩으로 계산합니다.",
        "",
        "## 반경과 후보 정책 비교 (ONNX, 단일 스레드)",
        "",
        "| 구성 | 인코더 | 후보 중앙값 | 정답 보존율 | 확장 요청 | 최대 반경(km) | 전체 p50 / p99 (ms) | NDCG@10 | HR@10 | 오프라인 기준 |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['variant']} | {r['backend']} | {r['n_candidates_p50']:.0f} | {r['stage1_recall']:.1%} | "
          f"{r['expanded_requests']} | {r['max_radius_used_km']:g} | {ms(r['total_p50'])} / {ms(r['total_p99'])} | "
          f"{r['ndcg@10']:.4f} | {r['hr@10']:.4f} | {r['offline']} |" for r in pipeline_rows],
        "",
        f"고정 반경 FP32의 E4 대조 결과는 **{'통과' if parity['passed'] else '허용 오차 초과'}**입니다. "
        f"NDCG@10 차이는 {parity['delta']['ndcg@10']:+.2e}, HR@10 차이는 {parity['delta']['hr@10']:+.2e}, "
        f"절대 허용 오차는 {parity['atol']:g}입니다.",
        "반경 확장 행은 실제로 검색한 후보로 계산한 별도 결과입니다.",
        "",
    ]
    if sweep_rows:
        lines.extend([
            "## FAISS 후보 수 비교 (고정 반경, ONNX fp32 단일 스레드)",
            "",
            "| N | 정답 보존율 | 후보 축소 p50 (ms) | 전체 p50 (ms) | NDCG@10 |",
            "|---|---|---|---|---|",
            *[f"| {r['n_candidates']} | {r['stage1_recall']:.1%} | {ms(r['narrow_p50'])} | "
              f"{ms(r['total_p50'])} | {r['ndcg@10']:.4f} |" for r in sweep_rows],
            "",
            f"## 합성 임베딩의 후보 수별 검색 비용 (hidden {meta['hidden']})",
            "",
            "| 후보 수 | 전체 점수 + 상위 10 (ms) | FAISS IVF 상위 200 (nprobe 8, ms) |",
            "|---|---|---|",
            *[f"| {r['n']:,} | {ms(r['brute_p50'])} | {ms(r['ivf_top200_p50'])} |" for r in scaling],
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="서빙 벤치")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--serving-config", default=DEFAULT_SERVING_CONFIG)
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    ecfg, scfg = load_config(args.eval_config), load_config(args.serving_config)
    radius = resolve_radius(cfg, args.radius_km)
    sampler, seed = scfg["model"]["sampler"], scfg["model"]["seed"]
    art = artifacts_dir(scfg, cfg["city"], radius, sampler, seed)
    pcfg, bcfg = pipeline_settings(scfg), scfg["bench"]
    meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
    if (meta["city"], meta["radius_km"]) != (cfg["city"], radius):
        parser.error("서빙 아티팩트의 도시 또는 반경이 다릅니다. 같은 CITY·R로 make export를 실행하세요.")
    split, _, files = load_split_inputs(cfg)
    n_requests = len(split.test)
    evaluation = meta.get("evaluation")
    if evaluation and (evaluation["data_id"] != fingerprint(files) or evaluation["split_id"] != split_fingerprint(split)):
        parser.error("서빙 아티팩트의 학습 결과와 현재 벤치의 데이터 또는 분할이 다릅니다.")
    offline = json.loads((Path(meta["run"]) / "summary.json").read_text(encoding="utf-8"))["test"]
    sizes = {"onnx-fp32": meta["size_mb"]["encoder_fp32.onnx"], "onnx-int8": meta["size_mb"]["encoder_int8.onnx"]}
    torch_size = (Path(meta["run"]) / "model.pt").stat().st_size / 1e6

    skipped = []
    if pcfg["retrieval"] == "faiss":
        skipped = [backend for backend in bcfg["backends"] if backend.startswith("torch-")]
        if skipped:
            print("[bench] FAISS 설정은 ONNX 백엔드로 측정합니다. Torch와 FAISS의 OpenMP 런타임을 분리합니다.")
    elif "torch-mps" in bcfg["backends"]:
        from torch.backends import mps

        if not mps.is_available():
            skipped.append("torch-mps")
            print("[bench] 이 환경에서 MPS를 사용할 수 없어 torch-mps 측정을 생략합니다.")
    backend_rows = []
    for backend in bcfg["backends"]:
        if backend in skipped:
            continue
        for threads in ([1] if backend == "torch-mps" else bcfg["threads"]):
            row = {
                "backend": backend, "threads": "기본" if threads == 0 else threads,
                "retrieval": pcfg["retrieval"], "radius_mode": pcfg["radius_mode"],
                "size_mb": sizes.get(backend, torch_size),
                **isolated(requests_job, split.test, art, backend, threads, pcfg, bcfg["warmup"]),
            }
            backend_rows.append(row)
            print(f"[bench] {backend:<10} threads={row['threads']}: E2E p50 {row['total_p50']:.2f}ms p99 {row['total_p99']:.2f}ms, "
                  f"stage2 p50 {row['rerank_p50']:.2f}ms, NDCG@10 {row['ndcg@10']:.4f}")

    fixed = {**pcfg, "retrieval": "full", "radius_mode": "fixed"}
    expanded = {**fixed, "radius_mode": "expand"}
    variants = []
    for backend in ("onnx-fp32", "onnx-int8"):
        variants.append(("고정 반경 · 전체", backend, fixed, None, "E4"))
        variants.append(("반경 확장 · 전체", backend, expanded, None, None))
    variants.append(("전체 카탈로그", "onnx-fp32", fixed, 1e5, "E3"))
    diagnostics = bcfg.get("retrieval_diagnostics", False)
    if diagnostics:
        for retrieval in ("faiss", "numpy"):
            variants.append((f"고정 반경 · {retrieval}", "onnx-fp32", {**fixed, "retrieval": retrieval}, None, "E4"))

    pipeline_rows = []
    for name, backend, policy, radius_override, ref in variants:
        summary = isolated(requests_job, split.test, art, backend, 1, policy, bcfg["warmup"], radius_override)
        row = {
            "variant": name, "backend": backend, "policy": policy, "radius_override": radius_override,
            "offline": f"{ref} {offline['ndcg@10'][ref]:.4f}" if ref else "—", **summary,
        }
        pipeline_rows.append(row)
        print(f"[bench] {name} {backend} | 후보 중앙값 {row['n_candidates_p50']:.0f}, "
              f"정답 보존율 {row['stage1_recall']:.1%}, 확장 {row['expanded_requests']}건, NDCG@10 {row['ndcg@10']:.4f}")

    fixed_fp32 = pipeline_rows[0]
    delta = {metric: fixed_fp32[metric] - offline[metric]["E4"] for metric in ("ndcg@10", "hr@10")}
    parity = {"atol": E4_ATOL, "delta": delta, "passed": all(abs(d) <= E4_ATOL for d in delta.values())}
    print(f"[bench] E4 대조 {'통과' if parity['passed'] else '허용 오차 초과'} | "
          f"NDCG@10 {delta['ndcg@10']:+.2e}, HR@10 {delta['hr@10']:+.2e}")

    sweep_rows = []
    scaling = []
    if diagnostics:
        for n_cand in (50, 100, 200, 500, 1000):
            policy = {**fixed, "retrieval": "faiss", "n_candidates": n_cand}
            summary = isolated(requests_job, split.test, art, "onnx-fp32", 1, policy, bcfg["warmup"])
            sweep_rows.append({"n_candidates": n_cand, **summary})
        scaling = isolated(scoring_cost, meta["hidden"])

    out = Path(ecfg["results_dir"]) / "serving" / cfg["city"] / f"R{radius:g}" / f"{sampler}_s{seed}" / attempt_id("bench")
    with publish_directory(out) as staging:
        write_json(staging / "bench.json", {
            "meta": meta, "pipeline": pcfg, "bench": bcfg, "environment": environment_metadata(), "git": git_metadata(),
            "backends": backend_rows, "skipped_backends": skipped,
            "pipelines": pipeline_rows, "offline": offline, "e4_parity": parity, "n_sweep": sweep_rows, "scaling": scaling,
        })
        (staging / "bench.md").write_text(
            render(meta, backend_rows, pipeline_rows, sweep_rows, scaling, n_requests, pcfg, parity, skipped), encoding="utf-8",
        )
    print(f"[bench] → {out / 'bench.md'}")


if __name__ == "__main__":
    main()
