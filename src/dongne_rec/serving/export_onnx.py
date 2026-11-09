"""학습된 SASRec → 서빙 아티팩트 (ONNX 인코더 fp32·int8 + 아이템 임베딩 + 장소 메타).

python -m dongne_rec.serving.export_onnx --city NYC
결과: artifacts/{city}/R{r}/{sampler}_s{seed}/
  encoder_fp32.onnx, encoder_int8.onnx  시퀀스 (B, max_len) int64 → 유저 표현 (B, H)
  item_emb.npy                          (N+1, H) float32, 점수 = h · e_i
  items.parquet                         item_idx, venue_id, lat, lon, category
  meta.json                             설정, 변환 오차, 파일 크기
인코더만 ONNX로 내보내고 후보 점수 계산은 numpy로 해요 (후보 수가 요청마다 달라서).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dongne_rec.config import city_dir, load_config, resolve_radius
from dongne_rec.data.split import read_parquet_input
from dongne_rec.experiments import saved_runs
from dongne_rec.models.sasrec import SASRec, pad_left
from dongne_rec.serving.pipeline import artifacts_dir
from dongne_rec.train.trainer import model_name, run_dir


def load_trained(run: Path) -> SASRec:
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))["model"]
    state = torch.load(run / "model.pt", map_location="cpu")
    model = SASRec(state["item_emb.weight"].shape[0] - 1, **cfg)
    model.load_state_dict(state)
    return model.eval()


def check_sequences(n_items: int, max_len: int, n: int = 64, seed: int = 0) -> np.ndarray:
    """변환 검증용 시퀀스: 길이가 1부터 max_len을 넘는 것까지 섞어서."""
    rng = np.random.default_rng(seed)
    lengths = np.concatenate([[1, max_len, max_len + 20], rng.integers(1, max_len, n - 3)])
    return pad_left([rng.integers(1, n_items + 1, int(n)) for n in lengths], max_len)


def export_encoder(model: SASRec, path: Path) -> None:
    example = torch.from_numpy(check_sequences(model.n_items, model.max_len, n=4))
    torch.onnx.export(
        model, (example,), str(path), input_names=["seq"], output_names=["h"], dynamo=True, external_data=False,
        dynamic_shapes={"seq": {0: torch.export.Dim("batch", min=1, max=4096)}},
    )


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """MatMul 가중치 int8 동적 양자화 (활성값은 실행 시 양자화)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(fp32_path), str(pre))
        quantize_dynamic(str(pre), str(int8_path), weight_type=QuantType.QInt8)


def max_abs_diff(onnx_path: Path, model: SASRec, seq: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.from_numpy(seq)).numpy()
    return float(np.abs(session.run(None, {"seq": seq})[0] - ref).max())


def export_artifacts(run: Path, items: pd.DataFrame, out: Path, extra_meta: dict) -> dict:
    model = load_trained(run)
    if len(items) != model.n_items:
        raise ValueError("체크포인트와 장소 카탈로그의 크기가 다릅니다.")
    out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out / "encoder_fp32.onnx", out / "encoder_int8.onnx"
    export_encoder(model, fp32)
    quantize_int8(fp32, int8)
    np.save(out / "item_emb.npy", model.item_emb.weight.detach().numpy().astype(np.float32))
    items[["item_idx", "venue_id", "lat", "lon", "category"]].to_parquet(out / "items.parquet", index=False)

    seq = check_sequences(model.n_items, model.max_len)
    meta = {
        **extra_meta,
        "run": str(run),
        "n_items": model.n_items,
        "max_len": model.max_len,
        "hidden": model.hidden,
        "max_abs_diff": {"fp32": max_abs_diff(fp32, model, seq), "int8": max_abs_diff(int8, model, seq)},
        "size_mb": {p.name: p.stat().st_size / 1e6 for p in (fp32, int8, out / "item_emb.npy")},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def select_checkpoint(
    ecfg: dict, city: str, radius: float, sampler: str, seed: int, items_hash: str, explicit: Path | None,
) -> tuple[Path, dict]:
    if explicit:
        run = explicit
        manifest = run / "metadata.json"
        info = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        actual = (summary["city"], summary["radius_km"], summary.get("sampler"), summary.get("seed"))
        if actual != (city, radius, sampler, seed):
            raise ValueError("선택한 체크포인트의 도시·반경·샘플러·시드와 서빙 설정을 맞춰 주세요.")
        if info and info["evaluation"]["processed_files"]["items.parquet"] != items_hash:
            raise ValueError("선택한 체크포인트와 현재 장소 카탈로그의 지문이 다릅니다.")
    else:
        run, info = run_dir(ecfg, city, radius, sampler, seed), {}
        checkpoints = [
            (path, record) for path, record in saved_runs(Path(ecfg["results_dir"]), city, radius)
            if record["kind"] == "sasrec" and record["tag"] is None
            and (record["sampler"], record["seed"]) == (sampler, seed)
            and (path / "model.pt").is_file()
        ]
        if checkpoints:
            for path, record in reversed(checkpoints):
                if record["evaluation"]["processed_files"]["items.parquet"] == items_hash:
                    run, info = path, record
                    break
            else:
                raise ValueError("현재 장소 카탈로그의 지문과 일치하는 체크포인트가 없습니다.")
    if not (run / "model.pt").is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다. {run / 'model.pt'}")
    return run, info


def main() -> None:
    parser = argparse.ArgumentParser(description="SASRec → ONNX 서빙 아티팩트")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--serving-config", default="configs/serving.yaml")
    parser.add_argument("--city")
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--run", type=Path, help="내보낼 학습 실행 디렉터리")
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    ecfg, scfg = load_config(args.eval_config), load_config(args.serving_config)
    radius = resolve_radius(cfg, args.radius_km)
    sampler, seed = scfg["model"]["sampler"], scfg["model"]["seed"]
    items, items_hash = read_parquet_input(city_dir(cfg, "processed_dir") / "items.parquet")
    try:
        run, info = select_checkpoint(ecfg, cfg["city"], radius, sampler, seed, items_hash, args.run)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    out = artifacts_dir(scfg, cfg["city"], radius, sampler, seed)
    extra = {"city": cfg["city"], "radius_km": radius, "model": model_name(sampler), "seed": seed}
    if info:
        extra.update({"run_id": info["run_id"], "evaluation": info["evaluation"]})
    meta = export_artifacts(run, items, out, extra)
    print(f"[export] {meta['model']} seed {seed} → {out}")
    print(f"[export] max |onnx − torch|: fp32 {meta['max_abs_diff']['fp32']:.2e}, int8 {meta['max_abs_diff']['int8']:.2e}")
    print("[export] size: " + ", ".join(f"{k} {v:.1f}MB" for k, v in meta["size_mb"].items()))


if __name__ == "__main__":
    main()
