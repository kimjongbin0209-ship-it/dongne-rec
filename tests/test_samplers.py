import numpy as np

from dongne_rec.train.samplers import RandomSampler, user_item_matrix


def test_random_sampler_zeroes_padding_and_stays_in_item_range():
    train = [np.array([1, 2, 3]), np.array([4, 5])]
    sampler = RandomSampler(10, user_item_matrix(train, 10), np.random.default_rng(0))
    inp = np.array([[0, 1, 2], [0, 0, 4]])
    pos = np.array([[0, 2, 3], [0, 0, 5]])
    for _ in range(200):
        negs = sampler.sample(inp, pos, np.array([0, 1]))
        assert (negs[pos == 0] == 0).all()
        assert ((negs[pos > 0] >= 1) & (negs[pos > 0] <= 10)).all()


def test_random_sampler_resamples_history_collisions_once():
    # 유저 이력 {1,2,3} + 정답 9 → 1회 뽑기 충돌 확률 0.4, 1회 재샘플 후 남는 확률 0.4² = 0.16
    sampler = RandomSampler(10, user_item_matrix([np.array([1, 2, 3])], 10), np.random.default_rng(0))
    n = 20_000
    negs = sampler.sample(np.ones((n, 1), dtype=np.int64), np.full((n, 1), 9), np.zeros(n, dtype=np.int64))
    collide = np.isin(negs, [1, 2, 3, 9]).mean()
    assert 0.14 < collide < 0.18
