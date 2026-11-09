import json
from pathlib import Path

import pytest

from dongne_rec.config import load_config
from dongne_rec.data.preprocess import preprocess
from dongne_rec.eval.compare import load_results, select_runs
from dongne_rec.train.trainer import apply_overrides, run_dir, train_run


def test_apply_overrides_parses_yaml_values_without_touching_original():
    cfg = {"model": {"dropout": 0.3, "hidden": 64}, "train": {"windows": "all"}}
    out = apply_overrides(cfg, ["model.dropout=0.5", "model.hidden=128", "train.windows=last"])
    assert out == {"model": {"dropout": 0.5, "hidden": 128}, "train": {"windows": "last"}}
    assert cfg["model"]["dropout"] == 0.3


def test_apply_overrides_rejects_unknown_keys():
    with pytest.raises(KeyError):
        apply_overrides({"model": {"dropout": 0.3}}, ["model.droput=0.5"])


def test_legacy_run_paths_still_resolve():
    ecfg = {"results_dir": "results"}
    assert run_dir(ecfg, "NYC", 3, "random", 0) == Path("results/sasrec/NYC/R3/random_s0")
    assert run_dir(ecfg, "NYC", 3, "random", 0, tag="d05") == Path("results/tuning/sasrec/NYC/R3/d05_random_s0")


def test_train_preserves_retries_and_compare_keeps_sampling_ratios_separate(tmp_path, synthetic_raw):
    proc = tmp_path / "processed" / "TEST"
    proc.mkdir(parents=True)
    interactions, items, _, _ = preprocess(synthetic_raw, min_user=5, min_item=5, collapse=True)
    interactions.to_parquet(proc / "interactions.parquet", index=False)
    items.to_parquet(proc / "items.parquet", index=False)
    results_dir = tmp_path / "results"
    cfg = load_config("configs/data.yaml", city="TEST", processed_dir=str(proc.parent))
    ecfg = load_config("configs/eval.yaml", results_dir=str(results_dir), n_sampled_negatives=4)
    scfg = apply_overrides(load_config("configs/sasrec.yaml"), [
        "model.hidden=8", "model.max_len=8", "train.device=cpu",
        "train.epochs=1", "train.eval_every=1", "train.geo_ratio=0.25",
    ])
    legacy = run_dir(ecfg, "TEST", 2, "mixed", 0)
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text("legacy", encoding="utf-8")

    first = train_run(cfg, ecfg, scfg, "mixed", 0, radius_km=2)
    second = train_run(cfg, ecfg, apply_overrides(scfg, ["train.geo_ratio=0.75"]), "mixed", 0, radius_km=2)
    retry = train_run(cfg, ecfg, scfg, "mixed", 0, radius_km=2)
    paths = [Path(result["run_dir"]) for result in (first, second, retry)]
    assert len(set(paths)) == 3 and all((path / "model.pt").is_file() for path in paths)
    assert first["experiment_id"] == retry["experiment_id"] != second["experiment_id"]
    assert (legacy / "summary.json").read_text(encoding="utf-8") == "legacy"

    metadata = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in paths]
    assert len({m["evaluation"]["candidate_cache_key"] for m in metadata}) == 1
    assert metadata[0]["configs"]["data"]["geo"]["radius_km"] == 2
    assert {path for path, _ in select_runs(results_dir, "TEST", 2)} == {paths[1], paths[2]}
    selected = select_runs(results_dir, "TEST", 2, paths[:2])
    models = set(load_results(selected)["model"])
    assert models == {"SASRec-S2 (mixed) α=0.25", "SASRec-S2 (mixed) α=0.75"}
