"""README용 정적 그림 (밝은 배경 PNG).

색은 dataviz 기본 팔레트의 범주형 1~4번 슬롯을 고정 순서로 써요 (선 차트 인접 쌍 기준 검증된 순서).
강조 모델 4개만 색을 쓰고 나머지 베이스라인은 회색 맥락선. 3·4번 슬롯은 밝은 배경 대비가 3:1 미만이라
항상 직접 라벨을 같이 달아요.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CONTEXT = "#c3c2b7"
PROTOCOL_LABELS = {"E1": "E1\nglobal-sampled", "E2": "E2\ngeo-sampled", "E3": "E3\nfull", "E4": "E4\ngeo-full ★"}
SLICE_LABELS = {"E1": "E1 global-sampled", "E2": "E2 geo-sampled", "E3": "E3 full", "E4": "E4 geo-full ★",
                "E4 new": "E4 · new places", "E4 repeat": "E4 · repeat visits"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def focus_colors(models: list[str], focus: list[str]) -> dict[str, str]:
    """강조 모델은 고정 순서로 1~4번 슬롯, 나머지는 회색."""
    colors = {m: CONTEXT for m in models}
    colors.update({m: SERIES[i] for i, m in enumerate(focus[: len(SERIES)])})
    return colors


def protocol_ranks(ndcg: pd.DataFrame, focus: list[str], title: str, path: Path, short: dict[str, str]) -> None:
    """bump chart: x = 프로토콜, y = 순위(1등이 위). ndcg는 모델 × 프로토콜 평균."""
    ranks = ndcg.rank(ascending=False, method="min")
    colors = focus_colors(list(ndcg.index), focus)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    x = list(range(len(ndcg.columns)))
    order = [m for m in ndcg.index if m not in focus] + [m for m in focus if m in ndcg.index]  # 강조선이 위에
    for m in order:
        is_focus = m in focus
        ax.plot(x, ranks.loc[m], color=colors[m], lw=2 if is_focus else 1.2, marker="o", ms=8 if is_focus else 5,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3 if is_focus else 2)
    for col, xpos, dx, ha in ((ndcg.columns[0], x[0], -10, "right"), (ndcg.columns[-1], x[-1], 10, "left")):
        for rank, group in ranks[col].groupby(ranks[col]):  # 동점은 한 라벨로
            names = list(group.index)
            emphasized = any(n in focus for n in names)
            ax.annotate(" = ".join(short.get(n, n) for n in names), (xpos, rank), xytext=(dx, 0), textcoords="offset points",
                        va="center", ha=ha, fontsize=9, color=INK if emphasized else MUTED)
    ax.set_xticks(x, [PROTOCOL_LABELS.get(c, c) for c in ndcg.columns])
    ax.set_ylim(len(ndcg) + 0.5, 0.5)
    ax.set_xlim(-0.9, len(x) - 0.1)
    ax.set_title(title + "\nrank by NDCG@10, best at top", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def sampler_effects(effects: pd.DataFrame, colors: dict[str, str], title: str, path: Path) -> None:
    """forest plot: 행 = 지표 구간, 점 = S0 대비 NDCG@10 차이(×100), 선 = 95% 신뢰구간.

    절대 차이를 써요. 상대(%)로 그리면 기준값이 작은 '신규' 구간의 구간 폭이 다른 행을 눌러버려요.
    effects 컬럼: comparison, slice, diff, lo, hi
    """
    slices = list(dict.fromkeys(effects["slice"]))
    comps = list(dict.fromkeys(effects["comparison"]))
    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(slices) + 1.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    offsets = {c: (i - (len(comps) - 1) / 2) * 0.24 for i, c in enumerate(comps)}
    for c in comps:
        d = effects[effects["comparison"] == c].set_index("slice").reindex(slices)
        y = [i + offsets[c] for i in range(len(slices))]
        ax.hlines(y, d["lo"] * 100, d["hi"] * 100, color=colors[c], lw=2, zorder=2)
        ax.plot(d["diff"] * 100, y, "o", color=colors[c], ms=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label=c)
    ax.set_yticks(range(len(slices)), [SLICE_LABELS.get(s, s) for s in slices])
    ax.set_ylim(len(slices) - 0.5, -0.5)
    ax.set_xlabel("Δ NDCG@10 × 100 vs SASRec-S0 (random) — dot = mean, line = 95% bootstrap CI", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(comps), labelcolor=INK,
              handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
