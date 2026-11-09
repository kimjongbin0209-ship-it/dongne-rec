"""Send real loopback HTTP traffic to the serving API, one Uvicorn process per trial."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import httpx
import numpy as np
import pandas as pd
import yaml

from dongne_rec.config import load_config, resolve_radius
from dongne_rec.data.geo import item_coords
from dongne_rec.data.split import Queries, load_split_inputs, split_fingerprint
from dongne_rec.experiments import attempt_id, environment_metadata, git_metadata
from dongne_rec.serving.pipeline import DEFAULT_SERVING_CONFIG, artifacts_dir, pipeline_settings
from dongne_rec.storage import fingerprint, publish_directory, write_json

BACKEND = "onnx-int8"
POLICIES = ("full", "faiss")
LABELS = {"full": "동네 전체", "faiss": "FAISS 200"}
PERCENTILES = (50, 95, 99)
SAMPLE_INTERVAL_S = 0.1
TIMEOUT_S = 10


def requests_for(queries: Queries, coords: tuple[np.ndarray, np.ndarray], count: int,
                 max_len: int, radius: float, rng: np.random.Generator) -> list[dict]:
    lat, lon = coords
    return [
        {"recent_item_ids": queries.histories[i][-max_len:].tolist(), "k": 10, "radius_km": radius,
         "lat": float(lat[queries.last_items[i]]), "lon": float(lon[queries.last_items[i]])}
        for i in rng.choice(len(queries), count, replace=False)
    ]


@asynccontextmanager
async def server(client: httpx.AsyncClient, artifact: Path, config: Path, expected: dict):
    env = {**os.environ, "DONGNE_ARTIFACTS": str(artifact), "DONGNE_BACKEND": BACKEND,
           "DONGNE_SERVING_CONFIG": str(config)}
    # Keep the kernel-assigned port reserved until Uvicorn owns the listener.
    with TemporaryFile() as log:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            fd, port = listener.fileno(), listener.getsockname()[1]
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "dongne_rec.serving.api:create_app", "--factory",
                 "--workers", "1", "--loop", "asyncio", "--http", "h11", "--fd", str(fd), "--no-access-log"],
                env=env, pass_fds=(fd,), stdout=log, stderr=log,
            )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    health = (await client.get(f"{url}/health", timeout=0.5)).raise_for_status().json()
                except httpx.TransportError:
                    await asyncio.sleep(0.05)
                    continue
                if health["backend"] != BACKEND or health["pipeline"] != expected:
                    raise RuntimeError(f"Unexpected server settings: {health}")
                break
            else:
                log.seek(0)
                raise RuntimeError(f"Uvicorn did not start:\n{log.read().decode(errors='replace')}")
            yield proc.pid, f"{url}/recommend"
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


async def send(client: httpx.AsyncClient, url: str, payloads: Sequence[dict], concurrency: int) -> list[dict]:
    pending = iter(enumerate(payloads))
    rows = [{} for _ in payloads]

    async def worker():
        for idx, payload in pending:
            start = time.perf_counter_ns()
            result = {}
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
                debug = body["debug"]
                total, n_geo, n_candidates = debug["latency_ms"]["total"], debug["n_geo"], debug["n_candidates"]
                if not (math.isfinite(total) and total >= 0 and 0 <= n_candidates <= n_geo
                        and len(body["items"]) == min(payload["k"], n_candidates)):
                    raise ValueError("Invalid recommendation response")
                result = {"internal_ms": total, "n_geo": n_geo, "n_candidates": n_candidates}
            except httpx.HTTPStatusError as exc:
                result = {"error": f"http_{exc.response.status_code}"}
            except httpx.RequestError as exc:
                result = {"error": type(exc).__name__}
            except (KeyError, TypeError, ValueError):
                result = {"error": "invalid_response"}
            rows[idx] = {"request": idx, "http_ms": (time.perf_counter_ns() - start) / 1e6, **result}

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return rows


def process_usage(pid: int) -> tuple[float, float, int]:
    start = time.perf_counter()
    raw = subprocess.check_output(["ps", "-o", "time=", "-o", "rss=", "-p", str(pid)], text=True)
    elapsed_at = (start + time.perf_counter()) / 2
    cpu, rss_kib = raw.split()
    days, _, clock = cpu.rpartition("-")
    cpu_s = int(days or 0) * 86_400 + sum(float(t) * 60**i for i, t in enumerate(reversed(clock.split(":"))))
    return elapsed_at, cpu_s, int(rss_kib)


async def trial(config: Path, expected: dict, artifact: Path, warmup: list[dict], measured: list[dict],
                concurrency: int, repeat: int) -> tuple[list[dict], dict]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=TIMEOUT_S, trust_env=False) as client:
        async with server(client, artifact, config, expected) as (pid, url):
            warm = await send(client, url, warmup, concurrency)
            errors = Counter(r["error"] for r in warm if "error" in r)
            if errors:
                raise RuntimeError(f"Warm-up failed: {dict(errors)}")

            finished = asyncio.Event()
            first = process_usage(pid)
            samples = [first]

            async def sample():
                while not finished.is_set():
                    try:
                        await asyncio.wait_for(finished.wait(), SAMPLE_INTERVAL_S)
                    except TimeoutError:
                        samples.append(await asyncio.to_thread(process_usage, pid))

            sampling = asyncio.create_task(sample())
            start = time.perf_counter()
            try:
                rows = await send(client, url, measured, concurrency)
                elapsed = time.perf_counter() - start
            finally:
                finished.set()
                await sampling
            last = process_usage(pid)
    info = {"policy": expected["retrieval"], "concurrency": concurrency, "repeat": repeat}
    resources = {
        **info, "elapsed_s": elapsed, "cpu_s": last[1] - first[1], "sample_elapsed_s": last[0] - first[0],
        "peak_rss_mib": max(s[2] for s in [*samples, last]) / 1024, "resource_samples": len(samples) + 1,
    }
    return [{**info, **row} for row in rows], resources


def summarize(rows: list[dict], resources: list[dict]) -> dict:
    ok = [row for row in rows if "error" not in row]

    def quantiles(key):
        return dict(zip((f"p{p}" for p in PERCENTILES), np.percentile([r[key] for r in ok], PERCENTILES).tolist())) if ok else None

    elapsed = sum(r["elapsed_s"] for r in resources)
    return {
        "requests": len(rows), "successes": len(ok), "errors": dict(Counter(r["error"] for r in rows if "error" in r)),
        "error_rate": 1 - len(ok) / len(rows), "elapsed_s": elapsed, "success_rps": len(ok) / elapsed,
        "internal_ms": quantiles("internal_ms"), "http_ms": quantiles("http_ms"),
        "server_cpu_pct_one_core": 100 * sum(r["cpu_s"] for r in resources) / sum(r["sample_elapsed_s"] for r in resources),
        "server_peak_rss_mib": max(r["peak_rss_mib"] for r in resources),
        "geo_p50": float(np.median([r["n_geo"] for r in ok])) if ok else None,
        "candidates_p50": float(np.median([r["n_candidates"] for r in ok])) if ok else None,
        "narrowed_share": sum(r["n_candidates"] < r["n_geo"] for r in ok) / len(ok) if ok else None,
    }


def render_table(conditions: list[dict]) -> str:
    def latencies(row, key):
        values = row[key]
        return " / ".join(f"{v:.2f}" for v in values.values()) if values else "—"

    lines = [
        "| 동시성 | 후보 정책 | 추천 내부 p50 / p95 / p99 (ms) | HTTP p50 / p95 / p99 (ms) | 성공 건/초 | 오류율 | 서버 CPU | 서버 최대 RSS (MiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(f"| {row['concurrency']} | {LABELS[row['policy']]} | {latencies(row, 'internal_ms')} | "
                     f"{latencies(row, 'http_ms')} | {row['success_rps']:.1f} | {row['error_rate']:.2%} | "
                     f"{row['server_cpu_pct_one_core']:.1f}% | {row['server_peak_rss_mib']:.1f} |")
    return "\n".join(lines)


async def measure(args, artifact: Path, scfg: dict, warmup: list[dict], measured: list[dict]):
    rows, runs = [], []
    with TemporaryDirectory(prefix="dongne-http-") as tmp:
        configs = {}
        for policy in POLICIES:
            settings = {**pipeline_settings(scfg), "retrieval": policy, "n_candidates": 200, "radius_mode": "fixed"}
            path = Path(tmp) / f"{policy}.yaml"
            path.write_text(yaml.safe_dump({**scfg, "pipeline": settings}), encoding="utf-8")
            configs[policy] = path, settings
        for repeat in range(1, args.repeats + 1):
            order = POLICIES if repeat % 2 else POLICIES[::-1]
            for concurrency in args.concurrency:
                for policy in order:
                    path, settings = configs[policy]
                    events, resources = await trial(path, settings, artifact, warmup, measured, concurrency, repeat)
                    run = {**resources, **summarize(events, [resources])}
                    rows.extend(events)
                    runs.append(run)
                    print(f"{repeat}/{args.repeats} c={concurrency} {policy}: {run['successes']}/{len(measured)} "
                          f"OK, {run['success_rps']:.1f} req/s", flush=True)
    conditions = []
    for concurrency in args.concurrency:
        for policy in POLICIES:
            events = [r for r in rows if (r["policy"], r["concurrency"]) == (policy, concurrency)]
            group = [r for r in runs if (r["policy"], r["concurrency"]) == (policy, concurrency)]
            conditions.append({"policy": policy, "concurrency": concurrency, **summarize(events, group)})
    return rows, runs, conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="/recommend 실제 HTTP 부하 측정")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--serving-config", default=DEFAULT_SERVING_CONFIG)
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(*args.concurrency, args.requests, args.warmup, args.repeats) < 1 or len(set(args.concurrency)) != len(args.concurrency):
        parser.error("동시성은 서로 다른 양의 정수여야 하고, 요청·워밍업·반복 횟수는 1 이상이어야 합니다.")

    cfg, ecfg, scfg = load_config(args.config, city=args.city), load_config(args.eval_config), load_config(args.serving_config)
    city, radius = cfg["city"], resolve_radius(cfg, args.radius_km)
    model = scfg["model"]
    artifact = artifacts_dir(scfg, city, radius, model["sampler"], model["seed"]).resolve()
    meta = json.loads((artifact / "meta.json").read_text(encoding="utf-8"))
    split, items, files = load_split_inputs(cfg)
    catalog = pd.read_parquet(artifact / "items.parquet").sort_values("item_idx").reset_index(drop=True)
    if (meta["city"], meta["radius_km"]) != (city, radius) or not catalog.equals(
        items[catalog.columns].sort_values("item_idx").reset_index(drop=True)
    ):
        parser.error("서빙 아티팩트와 부하 요청의 도시·반경·장소 카탈로그가 일치해야 합니다.")
    split_id = split_fingerprint(split)
    if evaluation := meta.get("evaluation"):
        if evaluation["data_id"] != fingerprint(files) or evaluation["split_id"] != split_id:
            parser.error("서빙 아티팩트의 학습 결과와 벤치의 데이터 또는 분할이 다릅니다.")
    if args.requests > len(split.test) or args.warmup > len(split.valid):
        parser.error("측정 건수와 워밍업 건수는 각각 test와 valid 요청 수를 넘을 수 없습니다.")
    coords, rng = item_coords(catalog), np.random.default_rng(args.seed)
    measured = requests_for(split.test, coords, args.requests, meta["max_len"], radius, rng)
    warmup = requests_for(split.valid, coords, args.warmup, meta["max_len"], radius, rng)
    rows, runs, conditions = asyncio.run(measure(args, artifact, scfg, warmup, measured))
    env = environment_metadata()
    results = Path(ecfg["results_dir"]) / "serving" / city / f"R{radius:g}" / artifact.name / attempt_id("http")
    info = {
        "schema_version": 1, "model": meta, "backend": BACKEND, "git": git_metadata(), "environment": env,
        "artifact_sha256": {name: hashlib.sha256((artifact / name).read_bytes()).hexdigest()
                            for name in ("meta.json", "items.parquet", "item_emb.npy", "encoder_int8.onnx")},
        "inputs": vars(args), "data_sha256": files, "split_id": split_id,
        "test_requests_sha256": fingerprint(measured), "valid_requests_sha256": fingerprint(warmup),
        "pipeline": {**pipeline_settings(scfg), "n_candidates": 200, "radius_mode": "fixed"},
        "request_timeout_s": TIMEOUT_S, "rss_sample_interval_s": SAMPLE_INTERVAL_S,
        "conditions": conditions, "trials": runs,
    }
    with publish_directory(results) as stage:
        write_json(stage / "http.json", info)
        pd.DataFrame(rows).to_parquet(stage / "requests.parquet", index=False)
        (stage / "http.md").write_text(
            f"# /recommend HTTP 부하 측정 — {city}, 고정 {radius:g}km\n\n"
            f"{env['cpu']} · {env['platform']} · {meta['model']} seed {meta['seed']} · {BACKEND} · Uvicorn 워커 1개.\n\n"
            f"같은 호스트의 루프백 HTTP/1.1에서 회차마다 valid {args.warmup:,}건으로 워밍업한 뒤 "
            f"동일한 test {args.requests:,}건을 요청했습니다. 조건별로 {args.repeats}회 반복했습니다.\n\n"
            f"{render_table(conditions)}\n\n"
            "지연은 성공 요청을 합산해 백분위수를 구했습니다. 추천 내부는 API의 `debug.latency_ms.total`, "
            "HTTP는 클라이언트의 요청 시작부터 응답 JSON 파싱과 확인까지의 시간입니다. "
            "성공 처리량은 전체 성공 건수를 측정 시간의 합으로 나눴습니다.\n\n"
            "서버 CPU 100%는 논리 코어 하나를 뜻합니다. RSS는 서버 PID를 100ms 간격과 회차의 양 끝에서 "
            "샘플링한 최댓값입니다. 요청을 생성한 클라이언트의 CPU와 메모리는 집계에서 제외했습니다.\n\n"
            "[회차별 결과와 실행 환경](http.json) · [요청별 시간과 후보 수](requests.parquet)\n",
            encoding="utf-8",
        )
    print(results / "http.md")


if __name__ == "__main__":
    main()
