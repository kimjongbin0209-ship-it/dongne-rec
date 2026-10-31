import numpy as np
import pandas as pd

from dongne_rec.baselines.itemknn import ItemKNN
from dongne_rec.baselines.pop import GeoPop, MostPop, PersonalFreq
from dongne_rec.data.geo import GeoIndex, radius_neighbors_csr
from dongne_rec.data.split import leave_one_out
from dongne_rec.eval.protocols import evaluate, sample_candidates

TRAIN = [np.array([1, 2, 1, 3]), np.array([2, 1]), np.array([4, 1])]  # 빈도: 1→4, 2→2, 3→1, 4→1, 5→0


def order(scores: np.ndarray) -> list[int]:
    return np.argsort(-scores[1:], kind="stable").tolist()  # 아이템 1..N을 점수 내림차순으로 (0-based)


def test_mostpop_scores_train_counts():
    s = MostPop().fit(TRAIN, n_items=5).score([np.array([2])], np.array([2]))
    assert s[0].tolist() == [0, 4, 2, 1, 1, 0]


def test_geopop_ranks_in_radius_items_first_then_by_popularity():
    lat = np.array([np.nan, 40.0, 41.0, 40.001, 40.002, 41.001])  # 1, 3, 4가 서로 가깝고 2, 5가 멀리
    lon = np.full(6, -74.0)
    geo = GeoIndex(1.0, lat, lon, *radius_neighbors_csr(lat, lon, 1.0))
    s = GeoPop(geo).fit(TRAIN, n_items=5).score([np.array([3])], np.array([3]))[0]
    assert np.argsort(-s[1:], kind="stable")[:3].tolist() == [0, 2, 3]  # 1, 3, 4 (반경 안) 먼저
    assert s[2] > s[5]                                                 # 반경 밖은 인기순


def test_personalfreq_orders_by_frequency_then_recency_then_popularity():
    pf = PersonalFreq().fit(TRAIN, n_items=5)
    s = pf.score([np.array([3, 5, 3, 2, 5, 3]), np.array([5, 4])], np.array([3, 4]))
    assert np.argsort(-s[0][1:])[:3].tolist() == [2, 4, 1]   # 3(3회) > 5(2회) > 2(1회)
    assert s[0][1] > s[0][4]                                   # 이력 밖: 1(인기 4) > 4(인기 1)
    assert s[1][4] > s[1][5] > s[1][1]                         # 동률이면 최근(4) 먼저, 이력 밖은 그 뒤


def test_itemknn_uses_cooccurrence_and_optional_self_similarity():
    train = [np.array([1, 2]), np.array([1, 2]), np.array([1, 3]), np.array([4, 5])]
    knn = ItemKNN(topk=10, last_n=1, include_self=False).fit(train, n_items=5)
    s = knn.score([np.array([4, 1])], np.array([1]))[0]
    assert s[2] > s[3] > 0 and s[4] == s[5] == 0              # 최근 1개(1)의 이웃만: 2가 3보다 자주 같이 등장
    s_self = knn.configure(include_self=True).score([np.array([4, 1])], np.array([1]))[0]
    assert s_self[1] == s[1] + 1


def test_geopop_equals_mostpop_under_geo_filtered_protocols(synthetic_raw):
    """E4는 반경 안만 보므로 GeoPop의 반경 가산점이 의미 없어져 MostPop과 같아야 해요."""
    from dongne_rec.data.geo import item_coords
    from dongne_rec.data.preprocess import preprocess

    inter, items, _, _ = preprocess(synthetic_raw, min_user=5, min_item=5, collapse=True)
    split = leave_one_out(inter, n_items=len(items))
    lat, lon = item_coords(items)
    geo = GeoIndex(3.0, lat, lon, *radius_neighbors_csr(lat, lon, 3.0))
    cands = sample_candidates(split.test, geo, n_neg=5, rng=np.random.default_rng(0))

    res = {
        m.name: evaluate(m.fit(split.train, split.n_items), split.test, geo, cands).set_index(["query", "protocol"])
        for m in (MostPop(), GeoPop(geo))
    }
    e4 = pd.IndexSlice[:, "E4"]
    np.testing.assert_allclose(res["MostPop"].loc[e4, "ndcg@10"], res["GeoPop"].loc[e4, "ndcg@10"])
