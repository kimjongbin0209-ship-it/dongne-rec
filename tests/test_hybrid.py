import numpy as np
import pytest

from dongne_rec.analysis.hybrid import WEIGHTS, e4_pool, evaluate_e4, grid_e4
from dongne_rec.baselines.hybrid import fuse_ranks, reciprocal_ranks
from dongne_rec.data.geo import GeoIndex, radius_neighbors_csr
from dongne_rec.data.split import Queries


def test_fusion_endpoints_preserve_each_models_ties_and_order():
    personal = reciprocal_ranks(np.array([5.0, 5.0, 2.0]))
    sequential = reciprocal_ranks(np.array([1.0, 3.0, 2.0]))
    visited = np.array([True, False, True])
    only_personal = fuse_ranks(personal, sequential, visited, np.array([1.0, 1.0]))
    only_sequential = fuse_ranks(personal, sequential, visited, np.array([0.0, 0.0]))
    assert only_personal[0] == only_personal[1] > only_personal[2]
    assert np.argsort(-only_sequential).tolist() == [1, 2, 0]


def test_grid_and_test_share_history_based_ranking_and_cold_misses():
    lat = np.array([np.nan, 40.000, 40.001, 40.002, 41.0])
    lon = np.full(len(lat), -74.0)
    geo = GeoIndex(0.15, lat, lon, *radius_neighbors_csr(lat, lon, 0.15))
    q = Queries("valid", np.array([0, 1]), [np.array([1, 2]), np.array([1, 2])], np.array([1, 0]), np.ones(2))
    pool = e4_pool(q, geo)
    assert [c.tolist() for c in pool.items] == [[1, 3], [1, 3]]
    assert [v.tolist() for v in pool.visited] == [[True, False], [True, False]]

    # PF's first visited candidate and SASRec's first unvisited candidate tie at these weights.
    pf = [reciprocal_ranks(np.array([3.0, 1.0]))] * 2
    sas = [reciprocal_ranks(np.array([1.0, 3.0]))] * 2
    weights = np.array([1.0, 0.0])
    fused = [fuse_ranks(p, s, seen, weights) for p, s, seen in zip(pf, sas, pool.visited)]
    scored = evaluate_e4(pool, fused)
    tied_ndcg = (1 + 1 / np.log2(3)) / 2
    assert scored["ndcg@10"].tolist() == pytest.approx([tied_ndcg, 0.0])
    assert scored["in_candidates"].tolist() == [True, False]
    index = np.flatnonzero(np.all(WEIGHTS == weights, axis=1))[0]
    assert grid_e4(pool, pf, sas)[index] == pytest.approx(scored["ndcg@10"].mean())
