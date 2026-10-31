import numpy as np

from dongne_rec.eval.metrics import expected_metrics


def ndcg(rank: int) -> float:
    return 1 / np.log2(rank + 1)


def test_metrics_without_ties():
    m = expected_metrics([0, 9, 10], [0, 0, 0], ks=(10,))
    np.testing.assert_allclose(m["hr@10"], [1, 1, 0])
    np.testing.assert_allclose(m["ndcg@10"], [1, ndcg(10), 0])
    np.testing.assert_allclose(m["mrr"], [1, 1 / 10, 1 / 11])


def test_ties_are_averaged_over_possible_ranks():
    # 위에 8개, 동점 3개 → 정답 순위는 9, 10, 11, 12 중 하나 (각 1/4)
    m = expected_metrics([8], [3], ks=(10,))
    np.testing.assert_allclose(m["hr@10"], [2 / 4])
    np.testing.assert_allclose(m["ndcg@10"], [(ndcg(9) + ndcg(10)) / 4])
    np.testing.assert_allclose(m["mrr"], [(1 / 9 + 1 / 10 + 1 / 11 + 1 / 12) / 4])


def test_ties_match_random_tie_break_simulation():
    rng = np.random.default_rng(0)
    g, t = 4, 12
    ranks = g + 1 + rng.integers(0, t + 1, 200_000)
    m = expected_metrics([g], [t], ks=(10,))
    assert abs(m["hr@10"][0] - (ranks <= 10).mean()) < 0.005
    assert abs(m["ndcg@10"][0] - np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean()) < 0.005
