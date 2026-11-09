# 베이스라인 — TKY, R=2km

`make baselines`로 생성된 파일이에요. test 유저 2,293명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@2km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **41.2%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 15.9% (채운 개수 평균 38.1)
- 재방문 정답 비율: 71.9%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.3734 | 0.1387 | 0.0587 | 0.0768 |
| GeoPop | 0.4506 | 0.1437 | 0.0768 | 0.0768 |
| PersonalFreq | 0.7966 | 0.3344 | 0.3721 | 0.2184 |
| ItemKNN(L=all, decay=1, self=True) | 0.7874 | 0.3241 | 0.2924 | 0.1868 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.5300 | 0.2048 | 0.1051 | 0.1210 |
| GeoPop | 0.6472 | 0.2124 | 0.1210 | 0.1210 |
| PersonalFreq | 0.8419 | 0.3519 | 0.5495 | 0.2983 |
| ItemKNN(L=all, decay=1, self=True) | 0.8566 | 0.3537 | 0.4379 | 0.2704 |

## 프로토콜별 모델 순위 (NDCG)

- E1: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E4: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 2.69 | 0.76 |
| GeoPop | 3.14 | 1.00 |
| PersonalFreq | 2.38 | 1.70 |
| ItemKNN(L=all, decay=1, self=True) | 2.43 | 1.56 |

### 정답이 동네 안인 쿼리만 (41.2%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.3168 | 0.3365 | 0.0450 | 0.1864 | 0.9414 |
| GeoPop | 0.8183 | 0.3487 | 0.1864 | 0.1864 | 2.3466 |
| PersonalFreq | 0.8088 | 0.8115 | 0.3838 | 0.5300 | 0.9966 |
| ItemKNN(L=all, decay=1, self=True) | 0.8166 | 0.7863 | 0.3029 | 0.4533 | 1.0386 |

## 재방문 / 신규 정답별 NDCG@10

재방문 71.9% / 신규 28.1%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0750 | 0.0170 | 0.0936 | 0.0341 |
| GeoPop | 0.0936 | 0.0341 | 0.0936 | 0.0341 |
| PersonalFreq | 0.5177 | 0.0000 | 0.2982 | 0.0147 |
| ItemKNN(L=all, decay=1, self=True) | 0.4050 | 0.0047 | 0.2498 | 0.0259 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 27.2% / 1-24h 28.7% / >24h 44.1%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.1134 | 0.0845 | 0.0492 |
| GeoPop | 0.1134 | 0.0845 | 0.0492 |
| PersonalFreq | 0.3175 | 0.2150 | 0.1595 |
| ItemKNN(L=all, decay=1, self=True) | 0.2668 | 0.1848 | 0.1388 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=1, self=True | 0.7676 | 0.3132 | 0.2421 | 0.1740 |
| L=all, decay=0.8, self=True | 0.7416 | 0.2945 | 0.2377 | 0.1645 |
| L=5, decay=1, self=True | 0.6658 | 0.2693 | 0.2191 | 0.1520 |
| L=5, decay=0.8, self=True | 0.6653 | 0.2672 | 0.2131 | 0.1482 |
| L=all, decay=0.5, self=True | 0.7012 | 0.2630 | 0.1845 | 0.1376 |
| L=5, decay=0.5, self=True | 0.6480 | 0.2508 | 0.1809 | 0.1355 |
| L=all, decay=1, self=False | 0.7266 | 0.2782 | 0.1121 | 0.1206 |
| L=all, decay=0.8, self=False | 0.6913 | 0.2538 | 0.1144 | 0.1125 |
| L=5, decay=1, self=False | 0.6112 | 0.2331 | 0.1114 | 0.1077 |
| L=5, decay=0.8, self=False | 0.6122 | 0.2314 | 0.1139 | 0.1058 |
| L=all, decay=0.5, self=False | 0.6610 | 0.2314 | 0.1156 | 0.1015 |
| L=5, decay=0.5, self=False | 0.5999 | 0.2204 | 0.1148 | 0.1007 |
| L=1, decay=1, self=True | 0.4192 | 0.1725 | 0.0908 | 0.0823 |
| L=1, decay=1, self=False | 0.4192 | 0.1725 | 0.0908 | 0.0823 |
