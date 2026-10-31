from pathlib import Path

import pytest

from dongne_rec.train.trainer import apply_overrides, run_dir


def test_apply_overrides_parses_yaml_values_without_touching_original():
    cfg = {"model": {"dropout": 0.3, "hidden": 64}, "train": {"windows": "all"}}
    out = apply_overrides(cfg, ["model.dropout=0.5", "model.hidden=128", "train.windows=last"])
    assert out == {"model": {"dropout": 0.5, "hidden": 128}, "train": {"windows": "last"}}
    assert cfg["model"]["dropout"] == 0.3


def test_apply_overrides_rejects_unknown_keys():
    with pytest.raises(KeyError):
        apply_overrides({"model": {"dropout": 0.3}}, ["model.droput=0.5"])


def test_tagged_runs_go_to_tuning_dir_outside_comparison():
    ecfg = {"results_dir": "results"}
    assert run_dir(ecfg, "NYC", 3, "random", 0) == Path("results/sasrec/NYC/R3/random_s0")
    assert run_dir(ecfg, "NYC", 3, "random", 0, tag="d05") == Path("results/tuning/sasrec/NYC/R3/d05_random_s0")
