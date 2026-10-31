import numpy as np
import torch

from dongne_rec.models.sasrec import SASRec, SASRecScorer, attention_mask, pad_left
from dongne_rec.train.samplers import RandomSampler, user_item_matrix
from dongne_rec.train.trainer import bce_loss, make_windows


def test_pad_left_keeps_most_recent_items():
    assert pad_left([np.array([1, 2, 3, 4, 5]), np.array([7])], 3).tolist() == [[3, 4, 5], [0, 0, 7]]


def test_make_windows_last_only_vs_all_positions():
    train = [np.arange(1, 9)]  # 1..8
    inp, pos, users = make_windows(train, max_len=3, all_windows=False)
    assert inp.tolist() == [[5, 6, 7]] and pos.tolist() == [[6, 7, 8]] and users.tolist() == [0]

    inp, pos, _ = make_windows(train, max_len=3, all_windows=True)
    assert inp.tolist() == [[5, 6, 7], [2, 3, 4], [0, 0, 1]]
    assert pos.tolist() == [[6, 7, 8], [3, 4, 5], [0, 0, 2]]
    assert sorted(pos[pos > 0].tolist()) == list(range(2, 9))  # 첫 장소를 뺀 모든 위치가 정확히 한 번 정답


def test_attention_mask_is_causal_and_hides_padding_keys():
    allowed = attention_mask(torch.tensor([[0, 0, 3, 5]]))[0, 0]
    assert allowed.tolist() == [
        [True, False, False, False],   # padding query: 자기 자신만
        [False, True, False, False],
        [False, False, True, False],   # 첫 실제 장소: padding key는 못 봄
        [False, False, True, True],
    ]


def test_encoder_is_causal_and_scores_all_items():
    torch.manual_seed(0)
    model = SASRec(n_items=20, max_len=6, hidden=16, n_blocks=2, n_heads=2, dropout=0.0).eval()
    a = torch.tensor([[0, 0, 3, 5, 7, 9]])
    b = a.clone()
    b[0, -1] = 11
    with torch.no_grad():
        ea, eb = model.encode(a), model.encode(b)
    assert torch.allclose(ea[:, :-1], eb[:, :-1], atol=1e-6)  # 마지막 입력을 바꿔도 앞 위치는 그대로
    assert not torch.allclose(ea[:, -1], eb[:, -1])

    scores = SASRecScorer(model, torch.device("cpu")).score([np.array([3, 5]), np.array([1, 2, 3, 4, 5, 6, 7, 8])], np.array([5, 8]))
    assert scores.shape == (2, 21) and np.isfinite(scores).all()


def test_training_reduces_loss_on_learnable_toy_sequences():
    torch.manual_seed(0)
    train = [np.tile(np.arange(1, 9), 3) for _ in range(8)]  # 1→2→…→8→1 반복: 다음 = 현재 + 1
    inp, pos, users = make_windows(train, max_len=10, all_windows=True)
    sampler = RandomSampler(20, user_item_matrix(train, 20), np.random.default_rng(0))
    model = SASRec(20, max_len=10, hidden=16, n_blocks=1, n_heads=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        negs = sampler.sample(inp, pos, users)
        loss, _ = bce_loss(model, *(torch.from_numpy(x) for x in (inp, pos, negs)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
