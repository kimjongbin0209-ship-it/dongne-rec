import numpy as np

from dongne_rec.eval.stats import paired_bootstrap


def test_identical_models_have_zero_difference_and_high_p():
    a = np.random.default_rng(0).random(500)
    r = paired_bootstrap(a, a)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["p"] == 1.0


def test_clear_improvement_is_significant_and_ci_covers_true_shift():
    rng = np.random.default_rng(0)
    base = rng.random(2000)
    other = base + 0.05 + rng.normal(0, 0.1, 2000)
    r = paired_bootstrap(base, other)
    assert r["lo"] < 0.05 < r["hi"] and r["lo"] > 0 and r["p"] < 0.01
    assert np.isclose(r["rel"], r["diff"] / base.mean()) and r["rel_lo"] < r["rel"] < r["rel_hi"]


def test_pairing_removes_shared_per_query_noise():
    # 쿼리 난이도 편차가 커도 두 모델 차이가 일정하면 신뢰구간이 좁아야 해요
    rng = np.random.default_rng(0)
    difficulty = rng.random(300) * 10
    r = paired_bootstrap(difficulty, difficulty + 0.01)
    assert r["hi"] - r["lo"] < 1e-9
