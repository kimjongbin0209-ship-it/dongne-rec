import numpy as np

from dongne_rec.data.geo import haversine_km, radius_neighbors_csr
from dongne_rec.train.samplers import GeoSampler, MixedSampler, RandomSampler, user_item_matrix

N = 60


def line_coords(spacing_deg: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """item i가 위도 40 + i·spacing (약 111m 간격)에 일렬로."""
    return np.concatenate([[np.nan], 40 + np.arange(1, N + 1) * spacing_deg]), np.concatenate([[np.nan], np.full(N, -74.0)])


def batch(item: int, positive: int, n: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.full((n, 1), item), np.full((n, 1), positive), np.zeros(n, dtype=np.int64)


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


def test_geo_sampler_draws_from_radius_of_current_place():
    lat, lon = line_coords()
    indptr, indices = radius_neighbors_csr(lat, lon, 0.5)  # 반경 안 = ±4개
    sampler = GeoSampler(N, user_item_matrix([np.array([30])], N), np.random.default_rng(0), indptr, indices)

    negs = sampler.sample(*batch(item=30, positive=31))

    within = haversine_km(lat[30], lon[30], lat[negs], lon[negs]) <= 0.5
    # 반경 안 9개 중 2개(이력 30, 정답 31)와 겹침: 재샘플 후에도 겹칠 확률 (2/9)² ≈ 0.049 → 전역 대체,
    # 대체분 중 반경 안(겹침 제외 7/58)은 우연 → 기댓값 1 − 0.049 × 51/58 ≈ 0.957
    assert abs(within.mean() - 0.957) < 0.01
    assert set(np.unique(negs[within])) == {26, 27, 28, 29, 32, 33, 34}
    assert not np.isin(negs, [30, 31]).any()


def test_geo_sampler_falls_back_to_global_when_radius_has_only_history():
    lat, lon = line_coords()
    indptr, indices = radius_neighbors_csr(lat, lon, 0.05)  # 반경 안 = 자기 자신뿐 (= 이력)
    sampler = GeoSampler(N, user_item_matrix([np.array([30])], N), np.random.default_rng(0), indptr, indices)

    negs = sampler.sample(*batch(item=30, positive=31))

    assert not np.isin(negs, [30, 31]).any() and len(np.unique(negs)) == N - 2


def test_mixed_sampler_ratio_controls_share_of_geo_negatives():
    lat, lon = line_coords()
    indptr, indices = radius_neighbors_csr(lat, lon, 0.5)  # 반경 안 9개 / 전체 60개
    empty_history = user_item_matrix([np.array([], dtype=np.int64)], N)

    def share_within(ratio: float) -> float:
        sampler = MixedSampler(N, empty_history, np.random.default_rng(0), indptr, indices, geo_ratio=ratio)
        negs = sampler.sample(*batch(item=30, positive=59, n=20_000))
        return (haversine_km(lat[30], lon[30], lat[negs], lon[negs]) <= 0.5).mean()

    assert share_within(1.0) == 1.0
    assert abs(share_within(0.0) - 9 / 60) < 0.02
    assert abs(share_within(0.5) - (0.5 + 0.5 * 9 / 60)) < 0.02
