"""쿼리별 평가 결과(long format) → 요약 표와 마크다운."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dongne_rec.analysis.eda import GAP_BINS_H, GAP_LABELS
from dongne_rec.data.split import Queries
from dongne_rec.eval.protocols import PROTOCOL_NAMES, PROTOCOLS


def with_meta(per_query: pd.DataFrame, q: Queries) -> pd.DataFrame:
    out = per_query.copy()
    qi = out["query"].to_numpy()
    out["user_idx"] = q.users[qi]
    out["is_repeat"] = q.is_repeat[qi]
    out["is_cold"] = q.targets[qi] == 0
    out["gap_bucket"] = pd.cut(q.gap_h[qi], bins=GAP_BINS_H, labels=GAP_LABELS)
    return out


def pivot(per_query: pd.DataFrame, metric: str, columns: str = "protocol") -> pd.DataFrame:
    """모델(입력 순서 유지) × columns 평균."""
    order = list(dict.fromkeys(per_query["model"]))
    return per_query.pivot_table(index="model", columns=columns, values=metric, aggfunc="mean", observed=True).reindex(order)


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}", index_name: str = "모델", header: dict | None = None) -> str:
    header = header or {}
    cols = [header.get(c, str(c)) for c in df.columns]
    lines = [f"| {index_name} | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join("—" if pd.isna(v) else fmt.format(v) for v in row) + " |")
    return "\n".join(lines)


PROTOCOL_HEADER = {p: f"{p} {PROTOCOL_NAMES[p]}{' ★' if p == 'E4' else ''}" for p in PROTOCOLS}


def protocol_rankings(table: pd.DataFrame) -> list[str]:
    """프로토콜마다 모델 순위 — 평가 방식에 따라 순위가 뒤집히는지 보여줘요. 같은 값은 '='."""
    lines = []
    for p in table.columns:
        col = table[p].sort_values(ascending=False)
        text = col.index[0]
        for (_, prev), (name, v) in zip(col.iloc[:-1].items(), col.iloc[1:].items()):
            text += f" {'=' if np.isclose(prev, v, rtol=0, atol=1e-9) else '>'} {name}"
        lines.append(f"- {p}: {text}")
    return lines


def breakdown_tables(per_query: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    full = per_query[per_query["protocol"].isin(["E3", "E4"])].copy()
    full["group"] = full["protocol"] + " " + np.where(full["is_repeat"], "재방문", "신규")
    by_gap = per_query[per_query["protocol"] == "E4"]
    return {
        "repeat": pivot(full, metric, columns="group").reindex(columns=["E3 재방문", "E3 신규", "E4 재방문", "E4 신규"]),
        "gap": pivot(by_gap, metric, columns="gap_bucket"),
    }
