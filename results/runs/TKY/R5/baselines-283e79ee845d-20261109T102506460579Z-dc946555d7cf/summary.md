# 베이스라인 — TKY, R=5km

`make baselines`로 생성된 파일이에요. test 유저 2,293명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@5km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **58.8%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 0.8% (채운 개수 평균 14.1)
- 재방문 정답 비율: 71.9%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.3687 | 0.2039 | 0.0587 | 0.0687 |
| GeoPop | 0.3959 | 0.2039 | 0.0687 | 0.0687 |
| PersonalFreq | 0.7970 | 0.4754 | 0.3721 | 0.2751 |
| ItemKNN(L=all, decay=1, self=True) | 0.7868 | 0.4679 | 0.2924 | 0.2284 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.5283 | 0.2969 | 0.1051 | 0.1055 |
| GeoPop | 0.5717 | 0.2969 | 0.1055 | 0.1055 |
| PersonalFreq | 0.8419 | 0.5003 | 0.5495 | 0.3881 |
| ItemKNN(L=all, decay=1, self=True) | 0.8520 | 0.5083 | 0.4379 | 0.3341 |

## 프로토콜별 모델 순위 (NDCG)

- E1: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop
- E3: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E4: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 1.81 | 0.85 |
| GeoPop | 1.94 | 1.00 |
| PersonalFreq | 1.68 | 1.35 |
| ItemKNN(L=all, decay=1, self=True) | 1.68 | 1.28 |

### 정답이 동네 안인 쿼리만 (58.8%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.3357 | 0.3469 | 0.0567 | 0.1169 | 0.9677 |
| GeoPop | 0.6247 | 0.3469 | 0.1169 | 0.1169 | 1.8008 |
| PersonalFreq | 0.8063 | 0.8086 | 0.3835 | 0.4679 | 0.9972 |
| ItemKNN(L=all, decay=1, self=True) | 0.8116 | 0.7960 | 0.3099 | 0.3885 | 1.0196 |

## 재방문 / 신규 정답별 NDCG@10

재방문 71.9% / 신규 28.1%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0750 | 0.0170 | 0.0858 | 0.0250 |
| GeoPop | 0.0858 | 0.0250 | 0.0858 | 0.0250 |
| PersonalFreq | 0.5177 | 0.0000 | 0.3803 | 0.0062 |
| ItemKNN(L=all, decay=1, self=True) | 0.4050 | 0.0047 | 0.3125 | 0.0135 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 27.2% / 1-24h 28.7% / >24h 44.1%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.0898 | 0.0698 | 0.0550 |
| GeoPop | 0.0898 | 0.0698 | 0.0550 |
| PersonalFreq | 0.3503 | 0.2672 | 0.2338 |
| ItemKNN(L=all, decay=1, self=True) | 0.2807 | 0.2219 | 0.2003 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=1, self=True | 0.7643 | 0.4571 | 0.2421 | 0.2016 |
| L=all, decay=0.8, self=True | 0.7408 | 0.4362 | 0.2377 | 0.1952 |
| L=5, decay=1, self=True | 0.6640 | 0.3989 | 0.2191 | 0.1795 |
| L=5, decay=0.8, self=True | 0.6664 | 0.3980 | 0.2131 | 0.1763 |
| L=all, decay=0.5, self=True | 0.7051 | 0.3994 | 0.1845 | 0.1593 |
| L=5, decay=0.5, self=True | 0.6539 | 0.3793 | 0.1809 | 0.1563 |
| L=all, decay=1, self=False | 0.7250 | 0.4185 | 0.1121 | 0.1240 |
| L=all, decay=0.8, self=False | 0.6934 | 0.3885 | 0.1144 | 0.1201 |
| L=all, decay=0.5, self=False | 0.6654 | 0.3641 | 0.1156 | 0.1150 |
| L=5, decay=0.8, self=False | 0.6142 | 0.3554 | 0.1139 | 0.1146 |
| L=5, decay=1, self=False | 0.6106 | 0.3559 | 0.1114 | 0.1138 |
| L=5, decay=0.5, self=False | 0.6054 | 0.3428 | 0.1148 | 0.1132 |
| L=1, decay=1, self=True | 0.4211 | 0.2633 | 0.0908 | 0.0894 |
| L=1, decay=1, self=False | 0.4211 | 0.2633 | 0.0908 | 0.0894 |
