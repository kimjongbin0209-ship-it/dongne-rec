# 베이스라인 — NYC, R=5km

`make baselines`로 생성된 파일이에요. test 유저 1,083명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@5km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **62.2%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 6.1% (채운 개수 평균 43.0)
- 재방문 정답 비율: 65.8%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.1776 | 0.1038 | 0.0274 | 0.0308 |
| GeoPop | 0.2887 | 0.1088 | 0.0308 | 0.0308 |
| PersonalFreq | 0.7144 | 0.4559 | 0.3631 | 0.2765 |
| ItemKNN(L=all, decay=1, self=True) | 0.7159 | 0.4545 | 0.2847 | 0.2292 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.2958 | 0.1918 | 0.0563 | 0.0540 |
| GeoPop | 0.4482 | 0.1991 | 0.0540 | 0.0540 |
| PersonalFreq | 0.7438 | 0.4718 | 0.5291 | 0.3823 |
| ItemKNN(L=all, decay=1, self=True) | 0.7864 | 0.4801 | 0.4657 | 0.3501 |

## 프로토콜별 모델 순위 (NDCG)

- E1: ItemKNN(L=all, decay=1, self=True) > PersonalFreq > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E4: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 1.71 | 0.89 |
| GeoPop | 2.65 | 1.00 |
| PersonalFreq | 1.57 | 1.31 |
| ItemKNN(L=all, decay=1, self=True) | 1.58 | 1.24 |

### 정답이 동네 안인 쿼리만 (62.2%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.1555 | 0.1668 | 0.0153 | 0.0495 | 0.9326 |
| GeoPop | 0.4348 | 0.1748 | 0.0495 | 0.0495 | 2.4868 |
| PersonalFreq | 0.7317 | 0.7326 | 0.3901 | 0.4444 | 0.9988 |
| ItemKNN(L=all, decay=1, self=True) | 0.7407 | 0.7303 | 0.3043 | 0.3683 | 1.0143 |

## 재방문 / 신규 정답별 NDCG@10

재방문 65.8% / 신규 34.2%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0248 | 0.0323 | 0.0379 | 0.0171 |
| GeoPop | 0.0379 | 0.0171 | 0.0379 | 0.0171 |
| PersonalFreq | 0.5515 | 0.0000 | 0.4160 | 0.0077 |
| ItemKNN(L=all, decay=1, self=True) | 0.4325 | 0.0000 | 0.3456 | 0.0050 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 13.1% / 1-24h 28.7% / >24h 58.2%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.0624 | 0.0419 | 0.0181 |
| GeoPop | 0.0624 | 0.0419 | 0.0181 |
| PersonalFreq | 0.3988 | 0.3047 | 0.2351 |
| ItemKNN(L=all, decay=1, self=True) | 0.3034 | 0.2472 | 0.2036 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=1, self=True | 0.6865 | 0.4319 | 0.2605 | 0.2017 |
| L=5, decay=1, self=True | 0.6251 | 0.3997 | 0.2632 | 0.1995 |
| L=all, decay=0.8, self=True | 0.6670 | 0.4166 | 0.2581 | 0.1990 |
| L=5, decay=0.8, self=True | 0.6253 | 0.3995 | 0.2550 | 0.1940 |
| L=all, decay=0.5, self=True | 0.6398 | 0.3951 | 0.2317 | 0.1778 |
| L=5, decay=0.5, self=True | 0.6168 | 0.3870 | 0.2311 | 0.1773 |
| L=all, decay=1, self=False | 0.6699 | 0.4168 | 0.1752 | 0.1489 |
| L=5, decay=0.8, self=False | 0.5970 | 0.3751 | 0.1636 | 0.1407 |
| L=all, decay=0.8, self=False | 0.6424 | 0.3949 | 0.1619 | 0.1401 |
| L=5, decay=1, self=False | 0.5976 | 0.3762 | 0.1590 | 0.1379 |
| L=5, decay=0.5, self=False | 0.5901 | 0.3617 | 0.1587 | 0.1367 |
| L=all, decay=0.5, self=False | 0.6178 | 0.3738 | 0.1552 | 0.1329 |
| L=1, decay=1, self=True | 0.4728 | 0.2822 | 0.1211 | 0.1109 |
| L=1, decay=1, self=False | 0.4728 | 0.2822 | 0.1211 | 0.1109 |
