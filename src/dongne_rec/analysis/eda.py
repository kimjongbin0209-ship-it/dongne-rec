"""0일차 EDA: 주 반경 R 결정과 README 표 1을 위한 통계.

전처리 결과(data/processed/{city}) → results/eda/{city}/
  eda_report.md, eda_stats.json, fig_distance_cdf.png, fig_overview.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dongne_rec.config import city_dir, load_config  # noqa: E402
from dongne_rec.data.geo import build_ball_tree, count_within, haversine_km, item_coords  # noqa: E402

QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.99)
GAP_BINS_H = [-np.inf, 1, 24, np.inf]
GAP_LABELS = ["<=1h", "1-24h", ">24h"]
SEQ_LEN_THRESHOLDS = (50, 100, 200)


def quantiles(x) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    out = {f"p{round(q * 100)}": float(np.quantile(x, q)) for q in QUANTILES}
    out.update(mean=float(x.mean()), max=float(x.max()))
    return out


def rkey(r: float) -> str:
    return f"{r:g}"


def transitions(inter: pd.DataFrame, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    """유저별 연속 체크인 쌍 (prev → next). inter는 (user_idx, ts_utc) 순 정렬 상태여야 해요.

    is_test: 유저의 마지막 쌍 (LOO test 정답, prev = test 시점의 현재 위치)
    is_repeat: next가 그 유저가 이전에 방문한 적 있는 장소인지
    """
    user = inter["user_idx"].to_numpy()
    item = inter["item_idx"].to_numpy()
    ts = inter["ts_utc"].dt.tz_convert(None).to_numpy()
    is_last = np.append(user[1:] != user[:-1], True)
    seen_before = inter.duplicated(["user_idx", "item_idx"]).to_numpy()

    nxt = np.flatnonzero(user[1:] == user[:-1]) + 1
    prv = nxt - 1
    return pd.DataFrame({
        "user_idx": user[nxt],
        "prev_item": item[prv],
        "next_item": item[nxt],
        "dist_km": haversine_km(lat[item[prv]], lon[item[prv]], lat[item[nxt]], lon[item[nxt]]),
        "gap_h": (ts[nxt] - ts[prv]) / np.timedelta64(1, "h"),
        "is_repeat": seen_before[nxt],
        "is_test": is_last[nxt],
    })


def choose_radius(coverage: dict[float, float], pool_share: dict[float, float], choices: list[float], max_pool_share: float) -> float:
    """동네 후보 풀이 카탈로그의 max_pool_share 이하인 후보 중 coverage가 가장 높은 반경.

    반경을 키우면 coverage(정답이 동네 안에 있을 확률)는 오르지만 동네가 카탈로그 전체에 가까워져
    '동네 제약'이 제약 역할을 못 해요. 둘의 균형을 풀 크기 상한으로 잡아요. 만족하는 후보가 없으면 가장 작은 반경.
    """
    allowed = [r for r in choices if pool_share[r] <= max_pool_share]
    return max(allowed, key=lambda r: coverage[r]) if allowed else min(choices)


def analyze(inter: pd.DataFrame, items: pd.DataFrame, geo_cfg: dict) -> tuple[dict, dict]:
    """(eda_stats, 그림용 배열)을 돌려줘요."""
    lat, lon = item_coords(items)
    tr = transitions(inter, lat, lon)
    test = tr[tr["is_test"]]
    choices = sorted(geo_cfg["radius_choices_km"])
    radii = sorted(set(geo_cfg["candidate_radii_km"]) | set(choices))

    coverage = {r: {"all": float((tr["dist_km"] <= r).mean()), "test": float((test["dist_km"] <= r).mean())} for r in radii}

    tree = build_ball_tree(lat, lon)
    neighbors, at_query, pool_share = {}, {}, {}
    for r in radii:
        per_item = count_within(tree, lat[1:], lon[1:], r)
        at_query[r] = np.concatenate([[0], per_item])[test["prev_item"].to_numpy()]
        pool_share[r] = float(np.median(at_query[r]) / len(items))
        neighbors[rkey(r)] = {
            "per_item": quantiles(per_item),
            "at_test_query": quantiles(at_query[r]),
            "pool_share_median": pool_share[r],
            "csr_mb": float(per_item.sum() * 4 / 1e6),
        }

    max_pool_share = geo_cfg["max_pool_share"]
    radius = choose_radius({r: c["all"] for r, c in coverage.items()}, pool_share, choices, max_pool_share)
    i = choices.index(radius)
    sensitivity = [r for r in choices[max(i - 1, 0) : i + 2] if r != radius]

    tr["gap_bucket"] = pd.cut(tr["gap_h"], bins=GAP_BINS_H, labels=GAP_LABELS)
    gap = {
        str(b): {
            "share": float(len(g) / len(tr)),
            "coverage": {rkey(r): float((g["dist_km"] <= r).mean()) for r in choices},
            "repeat": float(g["is_repeat"].mean()),
        }
        for b, g in tr.groupby("gap_bucket", observed=True)
    }

    seq_len = inter.groupby("user_idx").size().to_numpy()
    categories = inter.merge(items[["item_idx", "category"]], on="item_idx")["category"].value_counts()

    stats = {
        "dataset": {
            "users": int(inter["user_idx"].nunique()),
            "items": int(len(items)),
            "interactions": int(len(inter)),
            "density": float(len(inter) / (inter["user_idx"].nunique() * len(items))),
            "categories": int(items["category"].nunique()),
            "transitions": int(len(tr)),
        },
        "radius": {
            "chosen_km": radius,
            "sensitivity_km": sensitivity,
            "rule": f"후보 {choices} 중 동네 후보 풀(test 질의 위치 median)이 카탈로그의 {max_pool_share:.0%} 이하인 반경에서 coverage 최대",
        },
        "distance_km": {"all": quantiles(tr["dist_km"]), "test": quantiles(test["dist_km"])},
        "coverage": {rkey(r): c for r, c in coverage.items()},
        "neighbors": neighbors,
        "time_gap_h": quantiles(tr["gap_h"]),
        "time_gap_buckets": gap,
        "repeat_ratio": {"all": float(tr["is_repeat"].mean()), "test": float(test["is_repeat"].mean())},
        "seq_len": {
            **quantiles(seq_len),
            **{f"share_over_{t}": float((seq_len > t).mean()) for t in SEQ_LEN_THRESHOLDS},
        },
        "top_categories": {k: int(v) for k, v in categories.head(10).items()},
    }
    arrays = {"transitions": tr, "seq_len": seq_len, "at_query": at_query}
    return stats, arrays


def _ecdf(ax, x, **kw) -> None:
    x = np.sort(np.asarray(x))
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)


def plot_distance_cdf(stats: dict, tr: pd.DataFrame, choices: list[float], city: str, path: Path) -> None:
    floor = 0.01  # 로그축용 하한 (10m)
    radius = stats["radius"]["chosen_km"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _ecdf(ax, np.maximum(tr["dist_km"], floor), label="all transitions")
    _ecdf(ax, np.maximum(tr.loc[tr["is_test"], "dist_km"], floor), label="test (last transition per user)", alpha=0.8)
    for r in choices:
        chosen = r == radius
        ax.axvline(r, color="C3" if chosen else "0.6", ls="-" if chosen else "--", lw=1.4 if chosen else 1)
        cov = stats["coverage"][rkey(r)]["all"]
        ax.text(r * 1.04, 0.04, f"{'R=' if chosen else ''}{r:g}km\n{cov:.0%}", fontsize=8, color="C3" if chosen else "0.4")
    ax.set_xscale("log")
    ax.set_xlim(floor, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("distance from previous check-in (km, log scale)")
    ax.set_ylabel("cumulative share")
    ax.set_title(f"{city}: how far is the next check-in?")
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview(stats: dict, arrays: dict, choices: list[float], city: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for r in choices:
        _ecdf(ax, np.maximum(arrays["at_query"][r], 1), label=f"R={r:g}km")
    ax.set_xscale("log")
    ax.set_xlabel("# venues within R of current location (test queries)")
    ax.set_ylabel("cumulative share")
    ax.set_title("in-radius candidate pool size")
    ax.legend(fontsize=9)

    ax = axes[1]
    seq_len = arrays["seq_len"]
    ax.hist(seq_len, bins=np.logspace(np.log10(seq_len.min()), np.log10(seq_len.max()), 40), color="C0", alpha=0.8)
    ax.axvline(50, color="C3", lw=1.2)
    ax.text(52, ax.get_ylim()[1] * 0.9, "max_len=50", color="C3", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("check-ins per user (log scale)")
    ax.set_ylabel("# users")
    ax.set_title("sequence length")

    ax = axes[2]
    gap_h = arrays["transitions"]["gap_h"].to_numpy()
    _ecdf(ax, np.maximum(gap_h, 1 / 60))
    for h, label in [(1, "1h"), (24, "24h")]:
        ax.axvline(h, color="0.6", ls="--", lw=1)
        ax.text(h * 1.08, 0.04, label, fontsize=8, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("hours between consecutive check-ins (log scale)")
    ax.set_ylabel("cumulative share")
    ax.set_title("time gap")

    for ax in axes:
        ax.grid(alpha=0.3, which="both", lw=0.5)
    fig.suptitle(city)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{x:.1%}"


def render_report(city: str, stats: dict, prep: dict, choices: list[float]) -> str:
    d, rad = stats["dataset"], stats["radius"]
    R = rad["chosen_km"]
    cov_R = stats["coverage"][rkey(R)]
    nb_R = stats["neighbors"][rkey(R)]
    rejected = [r for r in choices if r > R]
    lines = [
        f"# EDA — {city}",
        "",
        "`make eda`로 생성된 파일이에요. 수치를 고치려면 코드를 고치고 다시 생성하세요.",
        "",
        "## 결정",
        "",
        f"- **주 반경 R = {R:g}km** — 직전 장소에서 다음 장소가 반경 안에 들어오는 비율 {_pct(cov_R['all'])} "
        f"(test 정답 기준 {_pct(cov_R['test'])} = E4 HR의 상한), 동네 후보 풀은 카탈로그의 {_pct(nb_R['pool_share_median'])} (median)",
        f"- 규칙: {rad['rule']}",
        *[
            f"- {r:g}km는 coverage {_pct(stats['coverage'][rkey(r)]['all'])}로 더 높지만 동네 후보 풀이 카탈로그의 "
            f"{_pct(stats['neighbors'][rkey(r)]['pool_share_median'])} (median)라 동네 제약의 대비가 약해져서 제외"
            for r in rejected
        ],
        f"- 민감도 분석용 반경: {', '.join(f'{r:g}km' for r in rad['sensitivity_km'])}",
        f"- R={R:g}km에서 test 질의 위치 반경 내 장소 수 median {nb_R['at_test_query']['p50']:.0f}, "
        f"p90 {nb_R['at_test_query']['p90']:.0f} → E2(반경 내 negative 100개)는 대부분 반경 안에서 채워지고, "
        f"이웃 CSR은 {nb_R['csr_mb']:.1f}MB",
        f"- `max_len=50`: 유저의 {_pct(stats['seq_len']['share_over_50'])}가 50보다 긴 시퀀스 (median {stats['seq_len']['p50']:.0f})",
        f"- 재방문 정답 비율 {_pct(stats['repeat_ratio']['test'])} (test) → 재방문/신규 분리 리포트 필요",
        "",
        "## 표 1. 데이터 통계 (전처리 후)",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 유저 | {d['users']:,} |",
        f"| 장소 | {d['items']:,} |",
        f"| 체크인 | {d['interactions']:,} |",
        f"| 밀도 | {d['density']:.2%} |",
        f"| 카테고리 | {d['categories']:,} |",
        f"| 유저당 체크인 (median / mean) | {stats['seq_len']['p50']:.0f} / {stats['seq_len']['mean']:.1f} |",
        f"| 재방문 정답 비율 (전체 / test) | {_pct(stats['repeat_ratio']['all'])} / {_pct(stats['repeat_ratio']['test'])} |",
        f"| coverage@{R:g}km (전체 / test) | {_pct(cov_R['all'])} / {_pct(cov_R['test'])} |",
        f"| 기간 (UTC) | {prep['final']['period_utc'][0][:10]} ~ {prep['final']['period_utc'][1][:10]} |",
        "",
        "### 전처리 단계별 변화",
        "",
        "| 단계 | 체크인 | 유저 | 장소 |",
        "|---|---|---|---|",
        *[f"| {s['step']} | {s['rows']:,} | {s['users']:,} | {s['items']:,} |" for s in prep["steps"]],
        "",
        f"장소 대표 좌표(체크인 좌표 중앙값)와 개별 체크인 좌표 차이: median {prep['coord_jitter_km']['p50'] * 1000:.0f}m, "
        f"p99 {prep['coord_jitter_km']['p99'] * 1000:.0f}m, max {prep['coord_jitter_km']['max']:.1f}km",
        "",
        "## 직전 → 다음 체크인 거리",
        "",
        "![distance CDF](fig_distance_cdf.png)",
        "",
        "| | p25 | p50 | p75 | p90 | p99 |",
        "|---|---|---|---|---|---|",
        *[
            f"| {name} | " + " | ".join(f"{stats['distance_km'][key][p]:.2f}km" for p in ("p25", "p50", "p75", "p90", "p99")) + " |"
            for name, key in (("전체 전이", "all"), ("test 정답", "test"))
        ],
        "",
        "## 반경별 coverage — 다음 장소가 반경 안에 있는 비율",
        "",
        "| R | 전체 전이 | test 정답 (E4 HR 상한) | 동네 후보 풀 / 카탈로그 (median) |",
        "|---|---|---|---|",
        *[
            f"| {k}km{' ★' if float(k) == R else ''} | {_pct(c['all'])} | {_pct(c['test'])} "
            f"| {_pct(stats['neighbors'][k]['pool_share_median'])} |"
            for k, c in stats["coverage"].items()
        ],
        "",
        "## 반경 내 장소 수 (자기 자신 포함)",
        "",
        "| R | 장소 기준 median / p90 / max | test 질의 위치 기준 median / p90 / max | 이웃 CSR 크기 |",
        "|---|---|---|---|",
        *[
            f"| {k}km | {n['per_item']['p50']:.0f} / {n['per_item']['p90']:.0f} / {n['per_item']['max']:.0f} "
            f"| {n['at_test_query']['p50']:.0f} / {n['at_test_query']['p90']:.0f} / {n['at_test_query']['max']:.0f} "
            f"| {n['csr_mb']:.1f}MB |"
            for k, n in stats["neighbors"].items()
        ],
        "",
        "## 체크인 간 시간 간격별",
        "",
        "![overview](fig_overview.png)",
        "",
        "| 간격 | 비율 | 재방문 | " + " | ".join(f"coverage@{r:g}km" for r in choices) + " |",
        "|---|---|---|" + "---|" * len(choices),
        *[
            f"| {b} | {_pct(g['share'])} | {_pct(g['repeat'])} | " + " | ".join(_pct(g["coverage"][rkey(r)]) for r in choices) + " |"
            for b, g in stats["time_gap_buckets"].items()
        ],
        "",
        f"시간 간격 median {stats['time_gap_h']['p50']:.1f}h, p90 {stats['time_gap_h']['p90']:.1f}h",
        "",
        "## 시퀀스 길이",
        "",
        f"median {stats['seq_len']['p50']:.0f}, p90 {stats['seq_len']['p90']:.0f}, max {stats['seq_len']['max']:.0f} · "
        + ", ".join(f"{t}보다 긴 유저 {_pct(stats['seq_len'][f'share_over_{t}'])}" for t in SEQ_LEN_THRESHOLDS),
        "",
        "## 상위 카테고리 (체크인 수)",
        "",
        "| 카테고리 | 체크인 |",
        "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in stats["top_categories"].items()],
        "",
    ]
    return "\n".join(lines)


def run(cfg: dict) -> dict:
    proc = city_dir(cfg, "processed_dir")
    inter = pd.read_parquet(proc / "interactions.parquet")
    items = pd.read_parquet(proc / "items.parquet")
    prep = json.loads((proc / "stats.json").read_text(encoding="utf-8"))

    stats, arrays = analyze(inter, items, cfg["geo"])
    choices = sorted(cfg["geo"]["radius_choices_km"])

    out = city_dir(cfg, "eda_dir")
    out.mkdir(parents=True, exist_ok=True)
    plot_distance_cdf(stats, arrays["transitions"], choices, cfg["city"], out / "fig_distance_cdf.png")
    plot_overview(stats, arrays, choices, cfg["city"], out / "fig_overview.png")
    (out / "eda_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "eda_report.md").write_text(render_report(cfg["city"], stats, prep, choices), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="0일차 EDA")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--city")
    args = parser.parse_args()

    cfg = load_config(args.config, city=args.city)
    stats = run(cfg)
    R = stats["radius"]["chosen_km"]
    current = cfg["geo"]["radius_km"]
    print(f"[eda] {cfg['city']} coverage (전체 / test):")
    for k, c in stats["coverage"].items():
        print(f"  R={k:>4}km  {c['all']:.1%} / {c['test']:.1%}{'  ← 주 반경' if float(k) == R else ''}")
    hint = "config와 일치" if current == R else f"config의 geo.radius_km({current})을 {R:g}로 바꾸고 make geo"
    print(
        f"[eda] 주 반경 R={R:g}km, 민감도 {stats['radius']['sensitivity_km']} ({hint}) "
        f"| 리포트: {city_dir(cfg, 'eda_dir') / 'eda_report.md'}"
    )


if __name__ == "__main__":
    main()
