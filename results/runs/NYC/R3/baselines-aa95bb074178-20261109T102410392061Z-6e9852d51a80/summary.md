# 베이스라인 — NYC, R=3km

`make baselines`로 생성된 파일이에요. test 유저 822명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@3km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **24.9%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 26.2% (채운 개수 평균 53.3)
- 재방문 정답 비율: 34.2%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.1101 | 0.0383 | 0.0175 | 0.0153 |
| GeoPop | 0.1550 | 0.0435 | 0.0153 | 0.0153 |
| PersonalFreq | 0.3822 | 0.1876 | 0.1854 | 0.1122 |
| ItemKNN(L=all, decay=0.8, self=True) | 0.3599 | 0.1679 | 0.1537 | 0.0998 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.1786 | 0.0669 | 0.0365 | 0.0262 |
| GeoPop | 0.2442 | 0.0777 | 0.0262 | 0.0262 |
| PersonalFreq | 0.4039 | 0.1971 | 0.2713 | 0.1545 |
| ItemKNN(L=all, decay=0.8, self=True) | 0.4101 | 0.1910 | 0.2440 | 0.1394 |

## 프로토콜별 모델 순위 (NDCG)

- E1: PersonalFreq > ItemKNN(L=all, decay=0.8, self=True) > GeoPop > MostPop
- E2: PersonalFreq > ItemKNN(L=all, decay=0.8, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=all, decay=0.8, self=True) > MostPop > GeoPop
- E4: PersonalFreq > ItemKNN(L=all, decay=0.8, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 2.88 | 1.14 |
| GeoPop | 3.56 | 1.00 |
| PersonalFreq | 2.04 | 1.65 |
| ItemKNN(L=all, decay=0.8, self=True) | 2.14 | 1.54 |

### 정답이 동네 안인 쿼리만 (24.9%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.1506 | 0.1535 | 0.0182 | 0.0615 | 0.9817 |
| GeoPop | 0.5047 | 0.1746 | 0.0615 | 0.0615 | 2.8909 |
| PersonalFreq | 0.7503 | 0.7522 | 0.3636 | 0.4499 | 0.9974 |
| ItemKNN(L=all, decay=0.8, self=True) | 0.7296 | 0.6734 | 0.3213 | 0.4004 | 1.0835 |

## 재방문 / 신규 정답별 NDCG@10

재방문 34.2% / 신규 65.8%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.0314 | 0.0102 | 0.0320 | 0.0067 |
| GeoPop | 0.0320 | 0.0067 | 0.0320 | 0.0067 |
| PersonalFreq | 0.5423 | 0.0000 | 0.3175 | 0.0055 |
| ItemKNN(L=all, decay=0.8, self=True) | 0.4473 | 0.0012 | 0.2834 | 0.0045 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 0.4% / 1-24h 9.4% / >24h 90.3%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.0000 | 0.0233 | 0.0146 |
| GeoPop | 0.0000 | 0.0233 | 0.0146 |
| PersonalFreq | 0.1111 | 0.1788 | 0.1053 |
| ItemKNN(L=all, decay=0.8, self=True) | 0.0000 | 0.1458 | 0.0955 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=0.8, self=True | 0.3771 | 0.1644 | 0.1503 | 0.0898 |
| L=5, decay=0.8, self=True | 0.3556 | 0.1577 | 0.1423 | 0.0894 |
| L=5, decay=1, self=True | 0.3597 | 0.1576 | 0.1447 | 0.0888 |
| L=all, decay=1, self=True | 0.3957 | 0.1736 | 0.1421 | 0.0861 |
| L=all, decay=0.5, self=True | 0.3569 | 0.1545 | 0.1262 | 0.0790 |
| L=5, decay=0.5, self=True | 0.3455 | 0.1508 | 0.1245 | 0.0782 |
| L=all, decay=1, self=False | 0.3806 | 0.1615 | 0.0956 | 0.0660 |
| L=5, decay=0.8, self=False | 0.3371 | 0.1437 | 0.0870 | 0.0604 |
| L=5, decay=0.5, self=False | 0.3287 | 0.1388 | 0.0856 | 0.0589 |
| L=all, decay=0.8, self=False | 0.3609 | 0.1514 | 0.0841 | 0.0583 |
| L=all, decay=0.5, self=False | 0.3397 | 0.1412 | 0.0819 | 0.0581 |
| L=5, decay=1, self=False | 0.3425 | 0.1441 | 0.0861 | 0.0575 |
| L=1, decay=1, self=True | 0.2688 | 0.1117 | 0.0686 | 0.0480 |
| L=1, decay=1, self=False | 0.2688 | 0.1117 | 0.0686 | 0.0480 |
