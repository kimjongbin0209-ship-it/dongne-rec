import numpy as np
import pandas as pd

from dongne_rec.data.split import leave_one_out


def make_inter(seqs: dict[int, list[tuple[int, float]]]) -> pd.DataFrame:
    """{user: [(item, 경과 시간[h]), ...]} → interactions (섞인 순서)."""
    t0 = pd.Timestamp("2012-04-03", tz="UTC")
    rows = [(u, i, t0 + pd.Timedelta(hours=h)) for u, seq in seqs.items() for i, h in seq]
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "ts_utc"]).sample(frac=1, random_state=0)


def test_leave_one_out_targets_histories_and_locations():
    inter = make_inter({0: [(1, 0), (2, 1), (3, 2), (4, 5), (5, 6)], 1: [(3, 0), (1, 1), (3, 3), (2, 27)]})
    split = leave_one_out(inter, n_items=5)

    assert [s.tolist() for s in split.train] == [[1, 2, 3], [3, 1]]
    assert [h.tolist() for h in split.valid.histories] == [[1, 2, 3], [3, 1]]
    assert split.valid.targets.tolist() == [4, 3]
    assert split.valid.last_items.tolist() == [3, 1]
    assert split.valid.is_repeat.tolist() == [False, True]
    assert split.valid.gap_h.tolist() == [3.0, 2.0]

    assert [h.tolist() for h in split.test.histories] == [[1, 2, 3, 4], [3, 1, 3]]
    assert split.test.targets.tolist() == [5, 2]
    assert split.test.last_items.tolist() == [4, 3]
    assert split.test.is_repeat.tolist() == [False, False]
    assert split.test.gap_h.tolist() == [1.0, 24.0]


def test_train_never_contains_valid_or_test_positions(synthetic_raw):
    from dongne_rec.data.preprocess import preprocess

    inter, items, _, _ = preprocess(synthetic_raw, min_user=5, min_item=5, collapse=True)
    split = leave_one_out(inter, n_items=len(items))
    seqs = [g["item_idx"].to_numpy() for _, g in inter.sort_values(["user_idx", "ts_utc"]).groupby("user_idx")]

    for seq, train, vh, th in zip(seqs, split.train, split.valid.histories, split.test.histories):
        np.testing.assert_array_equal(train, seq[:-2])
        np.testing.assert_array_equal(vh, seq[:-2])
        np.testing.assert_array_equal(th, seq[:-1])
    assert sum(len(t) for t in split.train) == len(inter) - 2 * len(seqs)
