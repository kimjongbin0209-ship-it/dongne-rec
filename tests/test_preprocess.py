import numpy as np
import pandas as pd

from conftest import make_raw
from dongne_rec.data.preprocess import collapse_consecutive_duplicates, filter_until_stable, load_raw, preprocess


def test_load_raw_parses_utc_and_local_time_and_utf8(tmp_path):
    path = tmp_path / "dataset_TSMC2014_NYC.csv"
    path.write_text(
        "userId,venueId,venueCategoryId,venueCategory,latitude,longitude,timezoneOffset,utcTimestamp\r\n"
        "470,49bbd6c0,4bf58dd8,Café,40.71981038,-74.00258103,-240,Tue Apr 03 18:00:09 +0000 2012\r\n"
        "1541,4f0fd5a8,4bf58dd9,Ramen /  Noodle House,35.70510109,139.61959,540,Tue Apr 03 18:17:18 +0000 2012\r\n",
        encoding="utf-8",
    )
    df = load_raw(path)

    assert df.loc[0, "category"] == "Café"
    assert df.loc[0, "ts_utc"] == pd.Timestamp("2012-04-03 18:00:09", tz="UTC")
    assert df.loc[0, "ts_local"] == pd.Timestamp("2012-04-03 14:00:09")
    assert df.loc[1, "ts_local"] == pd.Timestamp("2012-04-04 03:17:18")
    assert df["user_id"].tolist() == [470, 1541]


def test_collapse_drops_only_back_to_back_repeats_of_same_user():
    df = make_raw([(1, "A", 0), (1, "A", 1), (1, "B", 2), (1, "A", 3), (2, "A", 4)])
    out = collapse_consecutive_duplicates(df)
    assert list(zip(out["user_id"], out["venue_id"])) == [(1, "A"), (1, "B"), (1, "A"), (2, "A")]


def test_filter_recollapses_duplicates_created_by_kcore():
    # B는 1번만 방문돼서 k-core로 빠지고, 그러면 user 1의 A→B→A가 A→A가 돼서 다시 합쳐져야 해요.
    df = make_raw([(1, "A", 0), (1, "B", 1), (1, "A", 2), (1, "C", 3),
                   (2, "C", 0), (2, "A", 1), (2, "C", 2), (2, "A", 3)])
    out, _ = filter_until_stable(df, min_user=2, min_item=2, collapse=True)
    assert out.loc[out["user_id"] == 1, "venue_id"].tolist() == ["A", "C"]
    assert out.loc[out["user_id"] == 2, "venue_id"].tolist() == ["C", "A", "C", "A"]


def test_filter_invariants_on_random_data(synthetic_raw):
    df = synthetic_raw.sort_values(["user_id", "ts_utc"], kind="stable")
    out, _ = filter_until_stable(df, min_user=5, min_item=5, collapse=True)
    assert out.groupby("user_id").size().min() >= 5
    assert out.groupby("venue_id").size().min() >= 5
    same_user = out["user_id"].eq(out["user_id"].shift())
    assert not (same_user & out["venue_id"].eq(out["venue_id"].shift())).any()


def test_preprocess_ids_order_coords_and_category():
    coords = {"A": (40.70, -74.00), "B": (40.80, -73.90)}
    rows = [(u, v, h) for u in (10, 20) for h, v in enumerate("ABABAB")]
    df = make_raw(rows, coords)
    df.loc[0, "lat"] += 0.01                       # A의 한 체크인만 좌표가 튐 → 중앙값은 그대로
    df.loc[df["venue_id"] == "B", "category"] = ["Bar"] * 5 + ["Pub"]  # 최빈 카테고리 = Bar
    shuffled = df.sample(frac=1, random_state=0)

    inter, items, users, stats = preprocess(shuffled, min_user=2, min_item=2, collapse=True)

    assert items["item_idx"].tolist() == [1, 2] and items["venue_id"].tolist() == ["A", "B"]
    assert users["user_idx"].tolist() == [0, 1] and users["user_id"].tolist() == [10, 20]
    assert np.isclose(items.loc[0, "lat"], 40.70)
    assert items.loc[1, "category"] == "Bar"
    assert items["n_checkins"].tolist() == [6, 6]
    for _, g in inter.groupby("user_idx"):
        assert g["ts_utc"].is_monotonic_increasing
        assert g["item_idx"].tolist() == [1, 2, 1, 2, 1, 2]
    assert stats["final"]["interactions"] == len(inter) == 12
