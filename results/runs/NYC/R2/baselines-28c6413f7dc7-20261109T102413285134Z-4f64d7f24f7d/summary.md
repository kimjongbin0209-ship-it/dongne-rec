# 베이스라인 — NYC, R=2km

`make baselines`로 생성된 파일이에요. test 유저 1,083명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@2km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **40.8%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 32.4% (채운 개수 평균 63.4)
- 재방문 정답 비율: 65.8%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.1805 | 0.0724 | 0.0274 | 0.0536 |
| GeoPop | 0.3066 | 0.0968 | 0.0536 | 0.0536 |
| PersonalFreq | 0.7183 | 0.3068 | 0.3631 | 0.2052 |
| ItemKNN(L=all, decay=1, self=True) | 0.7195 | 0.3038 | 0.2847 | 0.1755 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.3013 | 0.1292 | 0.0563 | 0.0905 |
| GeoPop | 0.4359 | 0.1653 | 0.0905 | 0.0905 |
| PersonalFreq | 0.7476 | 0.3167 | 0.5291 | 0.2798 |
| ItemKNN(L=all, decay=1, self=True) | 0.7855 | 0.3181 | 0.4657 | 0.2621 |

## 프로토콜별 모델 순위 (NDCG)

- E1: ItemKNN(L=all, decay=1, self=True) > PersonalFreq > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E4: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 2.49 | 0.51 |
| GeoPop | 3.17 | 1.00 |
| PersonalFreq | 2.34 | 1.77 |
| ItemKNN(L=all, decay=1, self=True) | 2.37 | 1.62 |

### 정답이 동네 안인 쿼리만 (40.8%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.1624 | 0.1773 | 0.0194 | 0.1314 | 0.9159 |
| GeoPop | 0.6413 | 0.2371 | 0.1314 | 0.1314 | 2.7044 |
| PersonalFreq | 0.7497 | 0.7518 | 0.4031 | 0.5027 | 0.9971 |
| ItemKNN(L=all, decay=1, self=True) | 0.7578 | 0.7443 | 0.3042 | 0.4300 | 1.0182 |

## 재방문 / 신규 정답별 NDCG@10

재방문 65.8% / 신규 34.2%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0248 | 0.0323 | 0.0679 | 0.0261 |
| GeoPop | 0.0679 | 0.0261 | 0.0679 | 0.0261 |
| PersonalFreq | 0.5515 | 0.0000 | 0.3029 | 0.0167 |
| ItemKNN(L=all, decay=1, self=True) | 0.4325 | 0.0000 | 0.2618 | 0.0091 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 13.1% / 1-24h 28.7% / >24h 58.2%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.1189 | 0.0676 | 0.0320 |
| GeoPop | 0.1189 | 0.0676 | 0.0320 |
| PersonalFreq | 0.3513 | 0.2272 | 0.1614 |
| ItemKNN(L=all, decay=1, self=True) | 0.2928 | 0.1877 | 0.1430 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=1, self=True | 0.6834 | 0.2824 | 0.2605 | 0.1495 |
| L=5, decay=1, self=True | 0.6214 | 0.2633 | 0.2632 | 0.1485 |
| L=all, decay=0.8, self=True | 0.6647 | 0.2730 | 0.2581 | 0.1481 |
| L=5, decay=0.8, self=True | 0.6229 | 0.2620 | 0.2550 | 0.1453 |
| L=5, decay=0.5, self=True | 0.6147 | 0.2507 | 0.2311 | 0.1382 |
| L=all, decay=0.5, self=True | 0.6397 | 0.2542 | 0.2317 | 0.1370 |
| L=all, decay=1, self=False | 0.6679 | 0.2728 | 0.1752 | 0.1231 |
| L=all, decay=0.8, self=False | 0.6406 | 0.2559 | 0.1619 | 0.1192 |
| L=5, decay=0.8, self=False | 0.5943 | 0.2451 | 0.1636 | 0.1179 |
| L=5, decay=1, self=False | 0.5926 | 0.2467 | 0.1590 | 0.1176 |
| L=5, decay=0.5, self=False | 0.5884 | 0.2347 | 0.1587 | 0.1142 |
| L=all, decay=0.5, self=False | 0.6165 | 0.2390 | 0.1552 | 0.1098 |
| L=1, decay=1, self=True | 0.4739 | 0.1883 | 0.1211 | 0.0991 |
| L=1, decay=1, self=False | 0.4739 | 0.1883 | 0.1211 | 0.0991 |
