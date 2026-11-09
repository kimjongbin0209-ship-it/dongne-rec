# 베이스라인 — TKY, R=3km

`make baselines`로 생성된 파일이에요. test 유저 2,293명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@3km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **47.0%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 5.6% (채운 개수 평균 30.7)
- 재방문 정답 비율: 71.9%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.3707 | 0.1584 | 0.0587 | 0.0708 |
| GeoPop | 0.4171 | 0.1594 | 0.0708 | 0.0708 |
| PersonalFreq | 0.7959 | 0.3782 | 0.3721 | 0.2374 |
| ItemKNN(L=all, decay=1, self=True) | 0.7881 | 0.3712 | 0.2924 | 0.1992 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.5287 | 0.2283 | 0.1051 | 0.1095 |
| GeoPop | 0.5889 | 0.2305 | 0.1095 | 0.1095 |
| PersonalFreq | 0.8415 | 0.3958 | 0.5495 | 0.3271 |
| ItemKNN(L=all, decay=1, self=True) | 0.8555 | 0.4039 | 0.4379 | 0.2931 |

## 프로토콜별 모델 순위 (NDCG)

- E1: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E4: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 2.34 | 0.83 |
| GeoPop | 2.62 | 1.00 |
| PersonalFreq | 2.10 | 1.57 |
| ItemKNN(L=all, decay=1, self=True) | 2.12 | 1.47 |

### 정답이 동네 안인 쿼리만 (47.0%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.3209 | 0.3369 | 0.0492 | 0.1506 | 0.9527 |
| GeoPop | 0.7414 | 0.3391 | 0.1506 | 0.1506 | 2.1864 |
| PersonalFreq | 0.8011 | 0.8045 | 0.3817 | 0.5049 | 0.9958 |
| ItemKNN(L=all, decay=1, self=True) | 0.8096 | 0.7895 | 0.3063 | 0.4237 | 1.0254 |

## 재방문 / 신규 정답별 NDCG@10

재방문 71.9% / 신규 28.1%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0750 | 0.0170 | 0.0877 | 0.0275 |
| GeoPop | 0.0877 | 0.0275 | 0.0877 | 0.0275 |
| PersonalFreq | 0.5177 | 0.0000 | 0.3264 | 0.0098 |
| ItemKNN(L=all, decay=1, self=True) | 0.4050 | 0.0047 | 0.2685 | 0.0221 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 27.2% / 1-24h 28.7% / >24h 44.1%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.0991 | 0.0736 | 0.0515 |
| GeoPop | 0.0991 | 0.0736 | 0.0515 |
| PersonalFreq | 0.3238 | 0.2337 | 0.1864 |
| ItemKNN(L=all, decay=1, self=True) | 0.2672 | 0.1944 | 0.1603 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=1, self=True | 0.7659 | 0.3652 | 0.2421 | 0.1868 |
| L=all, decay=0.8, self=True | 0.7419 | 0.3446 | 0.2377 | 0.1782 |
| L=5, decay=1, self=True | 0.6669 | 0.3150 | 0.2191 | 0.1652 |
| L=5, decay=0.8, self=True | 0.6668 | 0.3135 | 0.2131 | 0.1610 |
| L=all, decay=0.5, self=True | 0.7032 | 0.3126 | 0.1845 | 0.1494 |
| L=5, decay=0.5, self=True | 0.6536 | 0.2982 | 0.1809 | 0.1465 |
| L=all, decay=1, self=False | 0.7261 | 0.3293 | 0.1121 | 0.1242 |
| L=all, decay=0.8, self=False | 0.6918 | 0.3019 | 0.1144 | 0.1173 |
| L=5, decay=1, self=False | 0.6151 | 0.2754 | 0.1114 | 0.1117 |
| L=5, decay=0.8, self=False | 0.6145 | 0.2753 | 0.1139 | 0.1104 |
| L=all, decay=0.5, self=False | 0.6644 | 0.2796 | 0.1156 | 0.1085 |
| L=5, decay=0.5, self=False | 0.6067 | 0.2642 | 0.1148 | 0.1074 |
| L=1, decay=1, self=True | 0.4197 | 0.2075 | 0.0908 | 0.0864 |
| L=1, decay=1, self=False | 0.4197 | 0.2075 | 0.0908 | 0.0864 |
