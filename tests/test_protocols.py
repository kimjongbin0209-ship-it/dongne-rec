import numpy as np
import pandas as pd
import pytest

from dongne_rec.data.geo import GeoIndex, haversine_km, radius_neighbors_csr
from dongne_rec.data.split import Queries, Split
from dongne_rec.eval.protocols import evaluate, load_or_sample_candidates, sample_candidates

N = 60


def line_geo(radius_km: float, spacing_deg: float = 0.001) -> GeoIndex:
    """item i가 위도 40 + i·spacing (약 111m 간격)에 일렬로 놓인 지도."""
    lat = np.concatenate([[np.nan], 40 + np.arange(1, N + 1) * spacing_deg])
    lon = np.concatenate([[np.nan], np.full(N, -74.0)])
    return GeoIndex(radius_km, lat, lon, *radius_neighbors_csr(lat, lon, radius_km))


def queries(histories: list[list[int]], targets: list[int]) -> Queries:
    return Queries("test", np.arange(len(targets)), [np.array(h) for h in histories], np.array(targets), np.ones(len(targets)))


class FixedScorer:
    name = "fixed"

    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def score(self, histories, last_items):
        return np.tile(self.scores, (len(last_items), 1))


def test_sampled_negatives_exclude_history_and_target_and_stay_in_radius():
    geo = line_geo(radius_km=3.0)  # 반경 안 ±27개
    q = queries([[5, 20, 30], [40, 41, 10]], [31, 12])
    c = sample_candidates(q, geo, n_neg=20, rng=np.random.default_rng(0))

    for i in range(len(q)):
        excluded = set(q.histories[i]) | {q.targets[i]}
        for p in ("E1", "E2"):
            negs = c[p][i]
            assert len(set(negs)) == 20 and not excluded & set(negs) and negs.min() >= 1 and negs.max() <= N
        last = q.last_items[i]
        assert (haversine_km(geo.lat[last], geo.lon[last], geo.lat[c["E2"][i]], geo.lon[c["E2"][i]]) <= 3.0).all()
    assert c["E2_fill"].tolist() == [0, 0]


def test_geo_negatives_are_filled_with_nearest_when_radius_is_too_small():
    geo = line_geo(radius_km=0.15)  # 반경 안 = 자기 자신 ±1
    q = queries([[1, 30]], [50])     # 반경 안 후보 29, 31 (30은 이력이라 제외)
    c = sample_candidates(q, geo, n_neg=4, rng=np.random.default_rng(0))
    assert c["E2_fill"].tolist() == [2]
    assert sorted(c["E2"][0].tolist()) == [28, 29, 31, 32]


def test_candidate_cache_rebuilds_when_target_changes_without_changing_query_count(tmp_path):
    cfg = {"processed_dir": str(tmp_path), "city": "TEST"}
    geo = line_geo(radius_km=3.0)
    valid = queries([[1, 2]], [3])
    split = Split(N, valid.histories, valid, queries([[1, 2, 3]], [4]))
    original = load_or_sample_candidates(cfg, split, geo, n_neg=10, seed=0)
    original_cache = next(tmp_path.rglob("*.npz"))

    target = original["test"]["E1"][0, 0]
    changed = Split(N, valid.histories, valid, queries([[1, 2, 3]], [target]))
    updated = load_or_sample_candidates(cfg, changed, geo, n_neg=10, seed=0)

    assert all(target not in updated["test"][protocol][0] for protocol in ("E1", "E2"))
    np.testing.assert_array_equal(load_or_sample_candidates(cfg, split, geo, 10, 0)["test"]["E1"], original["test"]["E1"])
    assert len(list(tmp_path.rglob("*.npz"))) == 2
    original_cache.write_bytes(b"incomplete cache")
    np.testing.assert_array_equal(load_or_sample_candidates(cfg, split, geo, 10, 0)["test"]["E1"], original["test"]["E1"])


def test_evaluate_excludes_current_place_and_handles_out_of_radius_target():
    geo = line_geo(radius_km=0.5)    # 반경 안 ±4개
    scores = np.zeros(N + 1)
    scores[30] = 100                 # 현재 장소: 점수는 가장 높지만 후보가 아님
    scores[31] = 50                  # 반경 안 정답
    scores[55] = 50                  # 반경 밖 정답
    q = queries([[10, 30], [10, 30]], [31, 55])
    cands = {"E1": np.array([[1, 2, 3], [1, 2, 3]]), "E2": np.array([[29, 32, 55], [29, 31, 32]])}

    res = evaluate(FixedScorer(scores), q, geo, cands).set_index(["query", "protocol"])

    assert res.loc[(0, "E3"), "mrr"] == pytest.approx(0.75)   # 현재 장소(30)는 빠지고 55와 동점 → 순위 1 또는 2
    assert res.loc[(0, "E4"), "ndcg@10"] == 1.0               # 55는 반경 밖이라 단독 1등
    assert res.loc[(0, "E2"), "hr@10"] == 1.0 and res.loc[(0, "E2"), "mrr"] == pytest.approx(0.75)  # 55와 동점
    assert res.loc[(1, "E3"), "mrr"] == pytest.approx(0.75)   # 31과 동점
    for p in ("E2", "E4"):                                     # 반경 밖 정답은 동네 프로토콜에서 miss
        assert res.loc[(1, p), "ndcg@10"] == 0.0 and not res.loc[(1, p), "in_candidates"]
    assert res.loc[(1, "E1"), "ndcg@10"] == 1.0


def test_evaluate_rejects_wrong_score_shape():
    geo = line_geo(radius_km=0.5)
    with pytest.raises(ValueError):
        evaluate(FixedScorer(np.zeros(N)), queries([[1, 2]], [3]), geo, {"E1": np.array([[4]]), "E2": np.array([[5]])})


def test_catalog_unknown_target_is_a_miss_in_every_protocol():
    geo = line_geo(radius_km=1)
    scores = np.zeros(N + 1)
    scores[0] = 100
    q = queries([[29, 30]], [0])
    cands = sample_candidates(q, geo, n_neg=5, rng=np.random.default_rng(0))
    out = evaluate(FixedScorer(scores), q, geo, cands)
    assert not out["in_candidates"].any()
    assert out[["ndcg@10", "hr@10", "mrr"]].eq(0).all().all()
