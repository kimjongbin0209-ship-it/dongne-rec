import numpy as np
import pandas as pd

from dongne_rec.data.geo import haversine_km, item_coords, load_geo_index, radius_neighbors_csr


def test_haversine_one_degree_of_latitude_is_about_111km():
    assert abs(haversine_km(40.0, -74.0, 41.0, -74.0) - 111.19) < 0.05


def test_haversine_is_zero_on_same_point_and_symmetric():
    assert haversine_km(40.7, -74.0, 40.7, -74.0) == 0
    assert np.isclose(haversine_km(40.7, -74.0, 35.7, 139.7), haversine_km(35.7, 139.7, 40.7, -74.0))


def test_radius_neighbors_csr_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 300
    items = pd.DataFrame({"item_idx": np.arange(1, n + 1), "lat": 40.7 + rng.normal(0, 0.03, n), "lon": -74 + rng.normal(0, 0.03, n)})
    lat, lon = item_coords(items)

    indptr, indices = radius_neighbors_csr(lat, lon, 2.0, chunk_size=64)  # 청크 경계도 같이 검사

    assert len(indptr) == n + 2
    assert indptr[0] == indptr[1] == 0  # item 0 = padding
    for i in range(1, n + 1):
        expected = np.flatnonzero(haversine_km(lat[i], lon[i], lat[1:], lon[1:]) <= 2.0) + 1
        np.testing.assert_array_equal(indices[indptr[i] : indptr[i + 1]], expected)


def test_geo_cache_rebuilds_after_catalog_coordinates_change(tmp_path):
    proc = tmp_path / "TEST"
    proc.mkdir()
    cfg = {"processed_dir": str(tmp_path), "city": "TEST"}
    items = pd.DataFrame({"item_idx": [1, 2], "lat": [40.0, 40.001], "lon": [-74.0, -74.0]})
    items.to_parquet(proc / "items.parquet", index=False)
    assert load_geo_index(cfg, 0.2).neighbors(1).tolist() == [1, 2]

    items.loc[1, "lat"] = 40.01
    items.to_parquet(proc / "items.parquet", index=False)
    assert load_geo_index(cfg, 0.2).neighbors(1).tolist() == [1]
