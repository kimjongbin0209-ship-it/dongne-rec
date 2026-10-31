import json

import numpy as np
import pandas as pd

from dongne_rec.analysis import eda
from dongne_rec.data import preprocess as prep


def test_transitions_marks_test_pair_and_repeats():
    inter = pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1],
        "item_idx": [1, 2, 1, 3, 1],
        "ts_utc": pd.to_datetime(["2012-04-03 00:00", "2012-04-03 01:00", "2012-04-03 03:00",
                                  "2012-04-03 00:00", "2012-04-04 00:00"], utc=True),
    })
    lat = np.array([np.nan, 40.0, 40.0, 41.0])
    lon = np.array([np.nan, -74.0, -74.0, -74.0])

    tr = eda.transitions(inter, lat, lon)

    assert list(zip(tr["prev_item"], tr["next_item"])) == [(1, 2), (2, 1), (3, 1)]
    assert tr["is_test"].tolist() == [False, True, True]
    assert tr["is_repeat"].tolist() == [False, True, False]  # user 1은 아이템 1을 처음 방문
    assert tr["gap_h"].tolist() == [1.0, 2.0, 24.0]
    assert np.isclose(tr["dist_km"].iloc[2], 111.19, atol=0.05)


def test_choose_radius_takes_highest_coverage_within_pool_limit():
    coverage = {1: 0.4, 2: 0.5, 3: 0.6, 5: 0.7}
    pool_share = {1: 0.02, 2: 0.05, 3: 0.12, 5: 0.34}
    assert eda.choose_radius(coverage, pool_share, [1, 2, 3, 5], max_pool_share=0.15) == 3
    assert eda.choose_radius(coverage, pool_share, [1, 2, 3, 5], max_pool_share=0.01) == 1  # 만족 후보 없음 → 최소 반경


def test_eda_run_end_to_end(tmp_path, synthetic_raw):
    cfg = {
        "city": "TEST",
        "processed_dir": str(tmp_path / "processed"),
        "eda_dir": str(tmp_path / "eda"),
        "geo": {"radius_km": 3, "candidate_radii_km": [0.5, 1, 2, 3, 5, 10], "radius_choices_km": [1, 2, 3, 5], "max_pool_share": 0.5},
    }
    inter, items, users, stats = prep.preprocess(synthetic_raw, min_user=5, min_item=5, collapse=True)
    out = tmp_path / "processed" / "TEST"
    out.mkdir(parents=True)
    inter.to_parquet(out / "interactions.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(stats))

    result = eda.run(cfg)

    eda_dir = tmp_path / "eda" / "TEST"
    for name in ("eda_report.md", "eda_stats.json", "fig_distance_cdf.png", "fig_overview.png"):
        assert (eda_dir / name).stat().st_size > 0
    assert result["radius"]["chosen_km"] in (1, 2, 3, 5)
    assert result["dataset"]["transitions"] == len(inter) - inter["user_idx"].nunique()
    coverages = [c["all"] for c in result["coverage"].values()]
    assert coverages == sorted(coverages)  # 반경이 커지면 coverage는 줄지 않음
