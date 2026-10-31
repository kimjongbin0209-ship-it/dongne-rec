# 베이스라인 — NYC, R=3km

`make baselines`로 생성된 파일이에요. test 유저 1,083명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@3km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **49.8%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 22.2% (채운 개수 평균 51.1)
- 재방문 정답 비율: 65.8%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.1811 | 0.0858 | 0.0274 | 0.0442 |
| GeoPop | 0.3024 | 0.0993 | 0.0442 | 0.0442 |
| PersonalFreq | 0.7134 | 0.3705 | 0.3631 | 0.2388 |
| ItemKNN(L=5, decay=1, self=True) | 0.6597 | 0.3412 | 0.2900 | 0.1933 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.3049 | 0.1530 | 0.0563 | 0.0783 |
| GeoPop | 0.4390 | 0.1755 | 0.0783 | 0.0783 |
| PersonalFreq | 0.7413 | 0.3818 | 0.5291 | 0.3269 |
| ItemKNN(L=5, decay=1, self=True) | 0.7130 | 0.3700 | 0.4523 | 0.2857 |

## 프로토콜별 모델 순위 (NDCG)

- E1: PersonalFreq > ItemKNN(L=5, decay=1, self=True) > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=5, decay=1, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=5, decay=1, self=True) > GeoPop > MostPop
- E4: PersonalFreq > ItemKNN(L=5, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 2.11 | 0.62 |
| GeoPop | 3.05 | 1.00 |
| PersonalFreq | 1.93 | 1.52 |
| ItemKNN(L=5, decay=1, self=True) | 1.93 | 1.50 |

### 정답이 동네 안인 쿼리만 (49.8%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.1669 | 0.1723 | 0.0192 | 0.0889 | 0.9688 |
| GeoPop | 0.5473 | 0.1995 | 0.0889 | 0.0889 | 2.7440 |
| PersonalFreq | 0.7392 | 0.7444 | 0.4001 | 0.4798 | 0.9931 |
| ItemKNN(L=5, decay=1, self=True) | 0.7090 | 0.6855 | 0.3071 | 0.3885 | 1.0343 |

## 재방문 / 신규 정답별 NDCG@10

재방문 65.8% / 신규 34.2%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0248 | 0.0323 | 0.0551 | 0.0233 |
| GeoPop | 0.0551 | 0.0233 | 0.0551 | 0.0233 |
| PersonalFreq | 0.5515 | 0.0000 | 0.3563 | 0.0123 |
| ItemKNN(L=5, decay=1, self=True) | 0.4390 | 0.0027 | 0.2869 | 0.0130 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 13.1% / 1-24h 28.7% / >24h 58.2%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.1002 | 0.0560 | 0.0258 |
| GeoPop | 0.1002 | 0.0560 | 0.0258 |
| PersonalFreq | 0.3980 | 0.2535 | 0.1956 |
| ItemKNN(L=5, decay=1, self=True) | 0.3299 | 0.1912 | 0.1636 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=5, decay=1, self=True | 0.6246 | 0.3123 | 0.2632 | 0.1702 |
| L=all, decay=1, self=True | 0.6875 | 0.3378 | 0.2605 | 0.1701 |
| L=all, decay=0.8, self=True | 0.6628 | 0.3249 | 0.2581 | 0.1685 |
| L=5, decay=0.8, self=True | 0.6264 | 0.3107 | 0.2550 | 0.1661 |
| L=5, decay=0.5, self=True | 0.6172 | 0.2985 | 0.2311 | 0.1532 |
| L=all, decay=0.5, self=True | 0.6405 | 0.3032 | 0.2317 | 0.1529 |
| L=all, decay=1, self=False | 0.6727 | 0.3273 | 0.1752 | 0.1345 |
| L=all, decay=0.8, self=False | 0.6408 | 0.3081 | 0.1619 | 0.1276 |
| L=5, decay=0.8, self=False | 0.6004 | 0.2909 | 0.1636 | 0.1247 |
| L=5, decay=1, self=False | 0.5987 | 0.2927 | 0.1590 | 0.1237 |
| L=5, decay=0.5, self=False | 0.5913 | 0.2765 | 0.1587 | 0.1204 |
| L=all, decay=0.5, self=False | 0.6194 | 0.2835 | 0.1552 | 0.1171 |
| L=1, decay=1, self=True | 0.4733 | 0.2174 | 0.1211 | 0.1011 |
| L=1, decay=1, self=False | 0.4733 | 0.2174 | 0.1211 | 0.1011 |
