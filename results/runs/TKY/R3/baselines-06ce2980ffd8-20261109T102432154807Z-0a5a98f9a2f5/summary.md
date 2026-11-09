# 베이스라인 — TKY, R=3km

`make baselines`로 생성된 파일이에요. test 유저 1,983명, E1/E2 negative 100개 (seed 0), 인기도·유사도는 train 구간만 사용, 동점은 무작위 순서 기댓값.

- coverage@3km (test 정답이 동네 안에 있는 비율 = E4 HR@10 상한): **21.5%**
- E2에서 반경 안 후보가 부족해 가까운 순으로 채운 쿼리: 10.9% (채운 개수 평균 34.6)
- 재방문 정답 비율: 47.5%

## NDCG@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.3415 | 0.0846 | 0.0722 | 0.0483 |
| GeoPop | 0.2643 | 0.0864 | 0.0483 | 0.0483 |
| PersonalFreq | 0.5741 | 0.1763 | 0.2212 | 0.1100 |
| ItemKNN(L=all, decay=1, self=True) | 0.5483 | 0.1644 | 0.1743 | 0.0927 |

## HR@10 (test)

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | 0.4664 | 0.1208 | 0.1266 | 0.0711 |
| GeoPop | 0.4066 | 0.1243 | 0.0711 | 0.0711 |
| PersonalFreq | 0.6268 | 0.1863 | 0.3389 | 0.1533 |
| ItemKNN(L=all, decay=1, self=True) | 0.6080 | 0.1781 | 0.2738 | 0.1357 |

## 프로토콜별 모델 순위 (NDCG)

- E1: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop > GeoPop
- E2: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > GeoPop > MostPop
- E3: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop > GeoPop
- E4: PersonalFreq > ItemKNN(L=all, decay=1, self=True) > MostPop = GeoPop

## 부풀림 배수 (NDCG)

E1 ÷ E2: 표준 평가(E1)가 동네 기준 샘플링 평가(E2)보다 몇 배 높게 나오는지. E3 ÷ E4: 동네 필터를 걸면 얼마나 떨어지는지 (1 미만이면 필터가 오히려 도움).

| 모델 | E1 ÷ E2 | E3 ÷ E4 |
|---|---|---|
| MostPop | 4.03 | 1.49 |
| GeoPop | 3.06 | 1.00 |
| PersonalFreq | 3.26 | 2.01 |
| ItemKNN(L=all, decay=1, self=True) | 3.33 | 1.88 |

### 정답이 동네 안인 쿼리만 (21.5%) — 반경 밖 miss를 빼고 negative 난이도 효과만 보기

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E1 ÷ E2 |
|---|---|---|---|---|---|
| MostPop | 0.3791 | 0.3931 | 0.0841 | 0.2245 | 0.9644 |
| GeoPop | 0.7865 | 0.4015 | 0.2245 | 0.2245 | 1.9589 |
| PersonalFreq | 0.8182 | 0.8186 | 0.3979 | 0.5107 | 0.9994 |
| ItemKNN(L=all, decay=1, self=True) | 0.8025 | 0.7636 | 0.3082 | 0.4305 | 1.0510 |

## 재방문 / 신규 정답별 NDCG@10

재방문 47.5% / 신규 52.5%

| 모델 | E3 재방문 | E3 신규 | E4 재방문 | E4 신규 |
|---|---|---|---|---|
| MostPop | 0.1334 | 0.0169 | 0.0945 | 0.0066 |
| GeoPop | 0.0945 | 0.0066 | 0.0945 | 0.0066 |
| PersonalFreq | 0.4662 | 0.0000 | 0.2293 | 0.0022 |
| ItemKNN(L=all, decay=1, self=True) | 0.3598 | 0.0068 | 0.1909 | 0.0040 |

## 직전 체크인과의 시간 간격별 NDCG@10 (E4)

<=1h 0.3% / 1-24h 7.5% / >24h 92.2%

| 모델 | <=1h | 1-24h | >24h |
|---|---|---|---|
| MostPop | 0.0000 | 0.0357 | 0.0495 |
| GeoPop | 0.0000 | 0.0357 | 0.0495 |
| PersonalFreq | 0.0000 | 0.1079 | 0.1104 |
| ItemKNN(L=all, decay=1, self=True) | 0.0578 | 0.0727 | 0.0944 |

## ItemKNN 튜닝 (valid ndcg@10, 기준 E4)

| 설정 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| L=all, decay=1, self=True | 0.5556 | 0.1910 | 0.1900 | 0.1043 |
| L=all, decay=0.8, self=True | 0.5260 | 0.1774 | 0.1784 | 0.0954 |
| L=5, decay=1, self=True | 0.4674 | 0.1633 | 0.1612 | 0.0895 |
| L=5, decay=0.8, self=True | 0.4660 | 0.1614 | 0.1584 | 0.0867 |
| L=all, decay=0.5, self=True | 0.4931 | 0.1576 | 0.1359 | 0.0758 |
| L=5, decay=0.5, self=True | 0.4543 | 0.1537 | 0.1342 | 0.0755 |
| L=all, decay=1, self=False | 0.5212 | 0.1644 | 0.0791 | 0.0608 |
| L=all, decay=0.8, self=False | 0.4864 | 0.1503 | 0.0813 | 0.0590 |
| L=5, decay=0.8, self=False | 0.4237 | 0.1363 | 0.0805 | 0.0564 |
| L=5, decay=1, self=False | 0.4255 | 0.1377 | 0.0808 | 0.0563 |
| L=all, decay=0.5, self=False | 0.4587 | 0.1361 | 0.0741 | 0.0509 |
| L=5, decay=0.5, self=False | 0.4164 | 0.1313 | 0.0749 | 0.0505 |
| L=1, decay=1, self=True | 0.2807 | 0.1019 | 0.0544 | 0.0416 |
| L=1, decay=1, self=False | 0.2807 | 0.1019 | 0.0544 | 0.0416 |
