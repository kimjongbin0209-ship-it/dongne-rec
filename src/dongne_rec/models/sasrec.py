"""SASRec (Kang & McAuley, ICDM 2018).

아이템 임베딩 + 학습형 위치 임베딩 → causal self-attention 블록 → 마지막 위치 hidden h,
score(i) = h · e_i (입출력 임베딩 공유). 입력은 왼쪽 패딩 고정 길이 max_len이라 마지막 위치가 항상
"가장 최근"이에요. 4일차 ONNX 변환을 위해 nn.MultiheadAttention 대신 어텐션을 직접 구현했어요.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def attention_mask(seq: torch.Tensor) -> torch.Tensor:
    """(B, L) → (B, 1, L, L) bool, True = query i가 key j를 볼 수 있음.

    causal(j ≤ i)이면서 key가 padding이 아닐 때만 허용. padding query 행이 전부 가려져 NaN이 나지 않게
    자기 자신은 항상 허용해요 (padding 위치 출력은 어차피 0으로 지워요).
    """
    length = seq.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).tril()
    eye = torch.eye(length, dtype=torch.bool, device=seq.device)
    return (causal & ((seq > 0).unsqueeze(1) | eye)).unsqueeze(1)


def pad_left(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    """시퀀스마다 최근 max_len개를 오른쪽 정렬, 앞은 0(padding)."""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for b, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s):
            out[b, -len(s):] = s
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads, self.head_dim = n_heads, hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """allowed: (B, 1, L, L) bool, True = query i가 key j를 볼 수 있음."""
        b, length, hidden = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~allowed, torch.finfo(att.dtype).min)
        att = self.dropout(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).reshape(b, length, hidden)
        return self.out(y)


class Block(nn.Module):
    """pre-LayerNorm Transformer 블록."""

    def __init__(self, hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), allowed))
        return x + self.dropout(self.ffn(self.ln2(x)))


class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int = 100, hidden: int = 64, n_blocks: int = 2, n_heads: int = 1, dropout: float = 0.3):
        super().__init__()
        self.n_items, self.max_len, self.hidden = n_items, max_len, hidden
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(hidden, n_heads, dropout) for _ in range(n_blocks))
        self.final_ln = nn.LayerNorm(hidden)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.xavier_normal_(p)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, L) 아이템 ID → (B, L, H). 위치 t의 출력은 t 이하 입력만 봐요."""
        length = seq.shape[1]
        valid = seq > 0
        x = self.item_emb(seq) * math.sqrt(self.hidden) + self.pos_emb(torch.arange(length, device=seq.device))
        x = self.emb_dropout(x) * valid.unsqueeze(-1)
        allowed = attention_mask(seq)
        for block in self.blocks:
            x = block(x, allowed) * valid.unsqueeze(-1)
        return self.final_ln(x)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, L) → 마지막 위치의 유저 표현 (B, H). 서빙·ONNX 변환 단위."""
        return self.encode(seq)[:, -1, :]

    def score_all(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, N+1) 전체 아이템 점수 (0번 열은 padding)."""
        return self.forward(seq) @ self.item_emb.weight.T


class SASRecScorer:
    """평가 하네스(eval.protocols.evaluate)용 래퍼."""

    def __init__(self, model: SASRec, device: torch.device, name: str = "SASRec"):
        self.model, self.device, self.name = model, device, name

    @torch.no_grad()
    def score(self, histories: list[np.ndarray], last_items: np.ndarray) -> np.ndarray:
        self.model.eval()
        seq = torch.from_numpy(pad_left(histories, self.model.max_len)).to(self.device)
        return self.model.score_all(seq).float().cpu().numpy()
