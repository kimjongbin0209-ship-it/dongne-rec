# 동네 반경 제약 순차 추천 + 경량 서빙 — 프로젝트 스펙

> 원본 플랜: `당근 ML 지원용 미니 프로젝트 플랜.pdf` (2026-10-31)
> 이 문서는 플랜을 **바로 구현에 들어갈 수 있는 수준**으로 구체화한 실행 스펙이에요.
> 목표 제출일: **2026-11-06(금) 전후**, 서류 마감 11/13.

---

## 0. 한눈에 보기

| 항목 | 결정 |
|---|---|
| 레포 이름 (제안) | `dongne-rec` (패키지 `dongne_rec`) |
| 데이터 | Foursquare NYC 체크인 (주), Tokyo는 시간 남으면 재현용 |
| 문제 | 최근 방문 시퀀스 + 현재 위치 → 다음 방문 장소 Top-10 |
| "동네" 정의 | 직전 방문 장소 좌표 기준 반경 **R = 3km** 원 (0일차 EDA로 확정, 민감도 분석 2km·5km) |
| 모델 | 베이스라인 4종(MostPop, GeoPop, ItemKNN, PersonalFreq) + SASRec |
| 핵심 가설 | **H1** 반경 내 negative 학습 효과, **H2** 평가 후보군에 따른 지표 부풀림 |
| 주 지표 | **E4 NDCG@10** (반경 필터 전체 랭킹) — 실제 서비스 화면과 가장 가까운 지표 |
| 서빙 | 지리 필터(BallTree) → FAISS 후보 → SASRec(ONNX fp32/int8) 재랭킹 |
| 환경 | MacBook (Apple Silicon), PyTorch MPS 학습 / CPU 서빙 벤치 |

### 최종적으로 README가 말해야 하는 3가지 주장

1. **"전역 랜덤 후보로 평가하면 성능이 X배 부풀려지고, 모델 순위까지 바뀐다."** (H2)
2. **"학습 negative를 동네 안에서 뽑으면 동네 기준 지표가 Y% 오른다 (그리고 이런 trade-off가 있다)."** (H1)
3. **"2단계 서빙으로 p99 Z ms, int8 양자화 시 정확도 −a% / 속도 +b%."** (서빙)

X, Y, Z, a, b를 채우는 게 이 프로젝트의 전부예요. 아래 모든 스펙은 이 숫자를 정직하게 만들기 위한 장치예요.

### 0일차 결과 (2026-10-31) — 이후 단계가 따라야 할 사실

상세: `results/eda/NYC/eda_report.md`, `results/eda/TKY/eda_report.md` (`make data eda geo`로 재생성)

| | NYC (주) | TKY |
|---|---|---|
| 전처리 후 유저 / 장소 / 체크인 | 1,083 / 9,786 / 163,649 | 2,293 / 14,960 / 454,288 |
| 밀도 | 1.54% | 1.32% |
| 유저당 체크인 median | 103 | 131 |
| 직전→다음 거리 median (전체 / test) | 1.92km / 3.06km | — |
| coverage@3km (전체 / test = E4 HR 상한) | 58.7% / 49.8% | 56.7% / 47.0% |
| 3km 동네 후보 풀 (test 위치 median, 카탈로그 대비) | 1,209 (12.4%) | 1,402 (9.4%) |
| 재방문 정답 비율 (test) | **65.8%** | **71.9%** |

여기서 나온 결정과 스펙 변경:

1. **R = 3km.** 처음 규칙("coverage ≈ 70%")대로면 5km인데, 5km에선 동네 후보 풀이 카탈로그의 34%(median)~49%(p90)라 반경 내 negative와 전역 negative가 비슷해져요. 규칙을 "동네 후보 풀 ≤ 카탈로그의 15%인 반경 중 coverage 최대"로 바꿨어요 (§2.3).
2. **재방문이 2/3를 넘어요.** PersonalFreq(본인 방문 빈도)를 선택이 아닌 필수 베이스라인으로 올렸고, E1/E2의 "이력 제외 negative"가 만드는 부풀림을 H2에 추가했어요 (§3.2, §4).
3. **`max_len` 50 → 100.** 유저의 97%가 50보다 길어서 원 SASRec처럼 유저당 마지막 `max_len`개만 학습하면 데이터 절반 이상을 버려요 (§5.1).
4. **직전 위치 신호는 시간 간격에 크게 좌우돼요.** coverage@3km가 1시간 이내 78%, 24시간 초과 45% → 결과를 시간 간격 구간별로도 리포트 (§3.2).
5. **test 정답은 전체 전이보다 멀어요** (median 3.06km vs 1.92km). 유저별 마지막 체크인이라는 선택 효과라서 README 한계에 적어요.

### 1일차 결과 (2026-10-31) — 베이스라인 test NDCG@10, R = 3km

상세: `results/baselines/{NYC,TKY}/R3/summary.md` (`make baselines`로 재생성)

| NYC | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ | E4 신규 정답 |
|---|---|---|---|---|---|
| MostPop | 0.181 | 0.086 | 0.027 | 0.044 | 0.023 |
| GeoPop | 0.302 | 0.099 | 0.044 | 0.044 | 0.023 |
| **PersonalFreq** | **0.713** | **0.371** | **0.363** | **0.239** | 0.012 |
| ItemKNN (최근 5개, 자기 유사도 포함) | 0.660 | 0.341 | 0.290 | 0.193 | 0.013 |

TKY도 순위와 경향이 같아요 (PersonalFreq E4 0.237, ItemKNN 0.199).

여기서 나온 사실과 결정:

1. **SASRec의 1차 관문은 PersonalFreq (NYC E4 0.239)**예요. 재방문이 66%라 "가 본 곳을 자주 간 순서로"만으로도 가장 강해요. SASRec 같은 next-item 모델은 재방문에 약하다고 알려져 있어서 못 넘을 수도 있어요. 그 경우 숨기지 말고 리포트해요.
2. **신규 정답 구간(E4 신규)이 진짜 승부처예요.** 모든 베이스라인이 NDCG 0.01~0.03 수준이에요. 당근 매물은 같은 물건을 다시 사지 않으니 **신규 구간이 당근 문제에 더 가까워요** → README에서 E4 전체와 함께 핵심 지표로 보고해요.
3. **H2 결과는 예상보다 미묘해요.** E1 ÷ E2 부풀림은 1.9~3.0배지만, 정답이 동네 안인 쿼리만 보면 지리를 안 쓰는 모델(MostPop, PersonalFreq, ItemKNN)은 E1 ≈ E2 (비율 0.97~1.03)예요. 즉 부풀림의 대부분은 ① 반경 밖 정답과 ② 이력 제외 negative(PersonalFreq E1 0.71 vs E3 0.36)에서 오고, **negative의 지리적 난이도 효과는 지리를 아는 모델(GeoPop 2.7배)에서만** 나타나요. SASRec은 공동 방문으로 위치를 암묵적으로 배우므로 어느 쪽인지가 2~3일차의 관전 포인트예요.
4. **E2 정의를 고쳤어요.** 처음 정의에서는 정답이 반경 밖이어도 정답으로 인정했어요. 그러면 negative가 전부 가까운 곳이라 "유일하게 먼 후보 = 정답"이라는 지름길이 생겨요. 지금은 E4처럼 miss로 처리해요 (§3.2).
5. NYC는 test 쿼리의 22%가 반경 안 후보 100개를 못 채워 가까운 순으로 채워요 (TKY 6%). 그 결과 E2에서 GeoPop이 MostPop보다 약간 높아요 (0.099 vs 0.086). 이건 버그가 아니라 채운 후보가 반경 밖이라서 생기는 차이예요.

---

## 1. 문제 정의

- 유저 u의 체크인 시퀀스 `s_1, s_2, …, s_T` (시간순), 각 장소 `i`는 좌표 `loc(i)`와 카테고리 `cat(i)`를 가짐
- 시점 t의 **현재 위치** `ℓ_t = loc(s_t)` (직전 방문 장소 좌표 — 미래 정보 누출 없음)
- **동네** `N_R(ℓ) = { i : haversine(loc(i), ℓ) ≤ R }`
- 과제: `s_1..s_t`와 `ℓ_t`가 주어졌을 때 `s_{t+1}`을 Top-K로 맞히기
- 당근 매물 추천과의 대응: 장소 ≈ 매물/가게, 체크인 ≈ 상호작용, 반경 R ≈ 동네 범위

---

## 2. 데이터 스펙

### 2.1 원천

- Kaggle "Foursquare NYC and Tokyo Check-ins" (원출처: Dingqi Yang et al., IEEE TSMC 2015 — README에 인용 필수, 비상업 연구용)
- 파일: `dataset_TSMC2014_NYC.csv`, `dataset_TSMC2014_TKY.csv`
- 컬럼: `userId, venueId, venueCategoryId, venueCategory, latitude, longitude, timezoneOffset, utcTimestamp`
- 규모(원본 확인 완료): NYC 227,428 체크인 / 1,083 유저 / 38,333 장소, TKY 573,703 / 2,293 / 61,858, 기간 2012-04-03 ~ 2013-02-16 (UTC), 결측 없음
- 인코딩은 **UTF-8** (latin-1로 읽으면 `Café`가 깨짐), 줄바꿈 CRLF
- 카테고리 이름 251개 / ID 400개 (같은 이름에 ID가 여러 개), 카테고리가 2개 이상 붙은 장소 337곳(NYC) → 최빈값으로 통일
- Kaggle API는 로그인 키가 필요하니 수동 다운로드 후 `data/raw/`에 둬요.

### 2.2 전처리 파이프라인 (`dongne_rec/data/preprocess.py`)

순서대로, 각 단계에서 제거된 행 수를 로그로 남겨요 (README 데이터 표에 그대로 씀).

1. `utcTimestamp` 파싱 (`"%a %b %d %H:%M:%S %z %Y"`), `timezoneOffset`(분)으로 현지 시각 컬럼 추가
2. 유저별 시간순 정렬
3. **연속 중복 제거**: 같은 유저가 같은 장소에 연속 체크인한 경우 첫 번째만 유지 ("다음 = 직전" 정답이 지표를 부풀리는 걸 방지)
4. **장소 좌표 대표값**: 같은 `venueId`의 위경도가 체크인마다 조금씩 달라서 중앙값으로 통일
5. **반복 k-core 필터**: 유저 ≥ 5, 장소 ≥ 5. k-core로 중간 체크인이 빠지면 연속 중복이 다시 생겨서(A→B→A에서 B 제거) 3번과 번갈아 변화가 없을 때까지 반복 (NYC·TKY 모두 4회에 수렴)
6. ID 재매핑: 아이템 `1..N` (venue_id 정렬 순, 0 = padding), 유저 `0..U-1`
7. 산출물 (`data/processed/{city}/`, `make data`)
   - `interactions.parquet` — `user_idx, item_idx, ts_utc, ts_local`
   - `items.parquet` — `item_idx, venue_id, lat, lon, category_id, category, n_checkins` (분할 기준 학습 빈도는 1일차 `split.py`에서 계산)
   - `users.parquet`, `stats.json` (단계별 감소, 좌표 노이즈)
   - `geo_neighbors_R{r}.npz` — 반경 R 이웃 CSR (`indptr, indices`, 자기 자신 포함, `make geo R=r`). R=3km: NYC 54MB, TKY 69MB

### 2.3 EDA 체크리스트 (`src/dongne_rec/analysis/eda.py` → `results/eda/{city}/`) — 결정을 내리기 위한 EDA

노트북 대신 스크립트로 만들어서 `make eda` 한 번으로 리포트·그림이 재생성돼요.

| 질문 | 산출물 | 이걸로 결정하는 것 |
|---|---|---|
| 직전→다음 체크인 거리 분포는? | 거리 CDF 그래프 | **주 반경 R** |
| R별로 다음 장소가 반경 안에 들어올 비율은? (R ∈ 0.5/1/2/3/5/10km) | coverage 표 | R, E4의 HR 상한 |
| R별 반경 내 장소 수 분포는? | 히스토그램 | geo sampler 메모리, Stage 1 후보 수 N |
| 다음 장소가 과거 방문 장소인 비율(재방문율)은? | 숫자 1개 | 재방문/신규 분리 리포트 필요성 |
| 시퀀스 길이 분포는? | 히스토그램 | SASRec `max_len` |
| 연속 체크인 간 시간 간격 분포는? | 히스토그램 | 시간 간격별 분석 구간 |

**R 결정 규칙 (0일차에 수정)**: 후보 1/2/3/5km 중 **동네 후보 풀(test 질의 위치의 반경 내 장소 수 median)이 카탈로그의 15% 이하인 반경에서 coverage가 가장 높은 것**을 주 반경으로, 바로 아래·위 값을 민감도 분석용으로 둬요. 반경을 키우면 정답이 동네 안에 있을 확률은 오르지만 동네가 카탈로그 전체에 가까워져 제약이 무의미해지기 때문이에요. 처음 규칙(coverage ≈ 70%)은 5km를 골랐는데 그때 동네 후보 풀이 카탈로그의 34%였어요. → **결과: R = 3km, 민감도 2km·5km (NYC·TKY 동일)**

---

## 3. 분할과 평가 프로토콜

### 3.1 분할 — 시간순 leave-one-out

- 유저별 마지막 1개 = test, 마지막에서 2번째 = valid, 나머지 = train
- valid 입력: `s_1..s_{T-2}`, 현재 위치 `loc(s_{T-2})` / test 입력: `s_1..s_{T-1}`, 현재 위치 `loc(s_{T-1})`
- 인기도·ItemKNN 유사도·SASRec 학습은 모두 **train 구간(각 유저의 마지막 2개 제외)만** 사용. test 때 valid 정답은 입력 이력으로만 쓰고 통계·학습에는 넣지 않아요 (SASRec이 재학습 없이 test하는 것과 같은 조건)
- 한계(README에 명시): LOO는 유저 간 전역 시간축 누출이 있어요 (다른 유저의 "미래" 체크인이 학습에 포함됨). 시간 남으면 전역 시간 기준 분할로 재검증.

### 3.2 평가 프로토콜 4종 — H2의 실험 장치

| ID | 이름 | 후보군 | 비고 |
|---|---|---|---|
| **E1** | Global-sampled | 정답 + **전역** 랜덤 negative 100개 | SASRec 원 논문 방식. negative는 유저 방문 이력 제외 |
| **E2** | Geo-sampled | 정답 + **반경 R 내** 랜덤 negative 100개 | 정답이 반경 밖이면 miss. 반경 내 부족 시 가까운 순으로 채움 |
| **E3** | Full | 전체 아이템 | 재방문 허용 (이력 마스킹 안 함) |
| **E4** | Geo-full ★ | 반경 R 내 전체 아이템 | 정답이 반경 밖이면 miss. **주 지표** |

- 구조: "샘플링 vs 전체" × "전역 vs 동네"의 2×2. 동네 프로토콜(E2/E4)은 반경 밖 정답을 miss로 둬요. 동네 필터가 있는 서비스에선 노출될 수 없는 장소이고, E2에서 정답으로 인정하면 "유일하게 먼 후보 = 정답"이라는 지름길이 생기기 때문이에요 (1일차에 발견해서 수정)
- **현재 장소(입력의 마지막 장소)는 모든 후보군에서 제외**: 연속 중복을 제거했으므로 정의상 정답이 될 수 없어요
- negative 난이도 효과만 따로 보려고 **정답이 동네 안인 쿼리만** 모은 표도 리포트 (반경 밖 miss 효과 제거)
- 지표: HR@10, NDCG@10 (정답 1개라 `NDCG = 1/log2(rank+1)` if rank ≤ 10), 보조로 MRR
- **동점 처리**: Pop 계열은 동점이 많아요. 정답 순위가 "정답보다 높은 수 + 1" ~ "+ 동점 수" 중 균등하게 나온다고 보고 **그 기댓값**을 계산해요 (`eval/metrics.py`). 무작위 추첨과 기댓값이 같고 시드 노이즈가 없어요. 낙관적 동점 처리는 흔한 버그예요
- 모든 결과를 **재방문 정답 / 신규 정답**으로 나눠서도 리포트 (test 정답의 66%(NYC)·72%(TKY)가 재방문)
- 주요 결과는 **직전 체크인과의 시간 간격 구간**(≤1h / 1–24h / >24h)별로도 리포트 — 직전 위치가 1시간 이내면 강한 신호, 하루 넘으면 약한 신호
- **이력 제외 negative의 함정**: E1/E2는 원 논문 관례대로 유저 방문 이력을 negative에서 빼요. 그러면 재방문 정답은 후보 101개 중 유일한 "가 본 곳"이 되어, 본인 방문 빈도만 보는 모델도 거의 다 맞혀요. 이건 지리와 별개의 부풀림이라 PersonalFreq로 드러내고 H2 결과에 같이 적어요 (E3/E4는 이력을 빼지 않아서 이 문제가 없음)
- E4와 함께 항상 **coverage@R**(정답이 반경 안에 있는 비율 = E4 HR의 상한)을 같이 표기
- E1/E2 sampled negative는 시드 고정 후 **모든 모델이 같은 후보 세트**를 쓰도록 미리 생성해 저장
- 참고: sampled metric의 불안정성 (Krichene & Rendle, KDD 2020) — E3/E4를 같이 두는 이유

### 3.3 통계

- SASRec 계열은 시드 3개(0/1/2) 평균 ± 표준편차
- 핵심 비교(H1: S1 vs S0 on E4)는 유저 단위 paired bootstrap(1,000회)으로 95% 신뢰구간

---

## 4. 베이스라인 스펙 (1일차)

| 모델 | 점수 정의 | 튜닝 (valid E4) |
|---|---|---|
| **MostPop** | `score(i) = train_count(i)` | 없음 |
| **GeoPop** | 반경 R 안: `train_count(i)`, 밖: `train_count(i) − ∞` (반경 밖은 맨 뒤로) | R |
| **ItemKNN** | 유저-아이템 이진 행렬의 아이템 간 cosine (top-100 이웃 유지), `score(i) = Σ_k λ^(k−1) · sim(s_{t−k+1}, i)`, 선택적으로 자기 유사도 1을 더해 재방문 반영 | 최근 L ∈ {1, 5, 전체}, λ ∈ {0.5, 0.8, 1.0}, 자기 유사도 포함 여부 → NYC 최근 5개·λ=1·포함, TKY 전체·λ=1·포함 |
| **PersonalFreq** | 유저 본인 방문 빈도 (동점은 최근 방문 우선), 방문 이력 밖은 MostPop 점수로 이어 붙임 | — |
| (선택) **Markov-1** | `P(s_{t+1}=i \| s_t)` 전이 카운트 | — |

> 예상 포인트: GeoPop은 E1에서 매우 강하게 나올 거예요 (전역 negative 대부분이 멀리 있어서 공짜로 이김). E2에서 그 이점이 사라지는 게 H2의 첫 증거예요. 재방문 정답이 2/3를 넘어서 PersonalFreq는 E3/E4에서도 강한 기준선이 될 거예요 — SASRec이 이걸 넘는지가 첫 번째 관문이에요.

---

## 5. SASRec 스펙 (2~3일차)

### 5.1 모델

- 구조: Kang & McAuley (ICDM 2018) 그대로 — 아이템 임베딩 + 학습형 위치 임베딩 → causal self-attention 블록 × 2 → 마지막 hidden `h`, `score(i) = h · e_i` (입출력 임베딩 공유)
- 어텐션은 ONNX 변환이 쉽도록 `nn.MultiheadAttention` 대신 직접 구현 (QKV linear + causal mask + softmax)
- 입력은 **왼쪽 패딩** 고정 길이 `max_len` (서빙 시 shape 고정을 위해)

| 하이퍼파라미터 | 기본값 | 비고 |
|---|---|---|
| `max_len` | 100 | 0일차 EDA: 유저당 체크인 median 103(NYC)·131(TKY), 97% 이상이 50 초과. 원 SASRec은 유저당 마지막 `max_len`개만 학습하므로, 그래도 짧으면 슬라이딩 윈도로 학습 샘플을 늘리기 |
| `hidden` | 64 | |
| `n_blocks` / `n_heads` | 2 / 1 | |
| `dropout` | 0.3 | 희소 데이터라 0.2~0.5 탐색 |
| `lr` / `batch` | 1e-3 / 128 | Adam |
| `epochs` | 최대 200, 5 epoch마다 valid 평가 | |
| early stopping | valid **E4 NDCG@10**, patience 4회(=20 epoch) | 모든 변형에 동일 기준 |
| loss | BCE, positive 1 : negative 1 | 원 논문 방식 |

### 5.2 Negative sampler — H1의 실험 장치 (`dongne_rec/train/samplers.py`)

각 학습 위치 t (입력 `s_t`, 정답 `s_{t+1}`)마다 negative 1개:

| ID | 이름 | 샘플링 분포 |
|---|---|---|
| **S0** | Random | 전체 아이템 균등 (원 SASRec) |
| **S1** | Geo | `N_R(loc(s_t))` 안에서 균등 — 반경 내가 비면 전역 랜덤으로 대체 |
| **S2** | Mixed | 확률 α로 Geo, 1−α로 Random (기본 α=0.5, 시간 되면 α ∈ {0.25, 0.5, 0.75} 스윕) |

- 공통: negative ≠ 정답, 유저 방문 이력에 있으면 1회 재샘플
- 구현: `geo_neighbors_R{r}.npz` CSR에서 벡터화 샘플링 — `deg = indptr[x+1]-indptr[x]; neg = indices[indptr[x] + floor(rand*deg)]`
- 기대 메커니즘 (README에 그대로 쓸 가설 문장):
  - SASRec은 좌표를 입력으로 받지 않아요. Random negative는 대부분 멀리 있는 장소라 모델이 주로 "가까움 vs 멂"을 배우고, 정작 **동네 안에서 무엇을 고를지**는 덜 배워요.
  - Geo negative는 동네 안 구분을 강제해서 E2/E4가 오를 것으로 예상.
  - 반대로 S1만 쓰면 "먼 곳 배제" 능력을 잃어 E1/E3는 떨어질 수 있음 → **S2가 양쪽을 다 잡는지**가 관전 포인트.
  - 리스크: 동네 안 negative 중 실제로 갈 만한 곳(false negative)이 섞임 → 결과가 기대와 다르면 이걸로 해석.
- (선택) 대조군: 전체 softmax CE loss — 아이템 1만 개 안팎이면 MPS에서 충분히 돌아가요. "샘플링 전략 효과 vs 전체 softmax 상한"을 보여줄 수 있어요.

### 5.3 실험 매트릭스

- 학습: {S0, S1, S2} × 시드 {0, 1, 2} = **9회** (각 수 분)
- 평가: 각 체크포인트 × {E1, E2, E3, E4} × {HR@10, NDCG@10}
- 민감도(시간 되면): 주 반경 R의 아래·위 값으로 S0 vs S1만 재실행

---

## 6. 결과 표/그림 템플릿 (README에 들어갈 형태)

**표 1. 데이터 통계** — 유저 수, 아이템 수, 체크인 수, 밀도, 평균 시퀀스 길이, 재방문율, coverage@R

**표 2. 프로토콜별 성능 (NDCG@10, test)**

| 모델 | E1 Global-sampled | E2 Geo-sampled | E3 Full | E4 Geo-full ★ |
|---|---|---|---|---|
| MostPop | | | | |
| GeoPop | | | | |
| ItemKNN | | | | |
| PersonalFreq | | | | |
| SASRec-S0 (random) | ±  | ± | ± | ± |
| SASRec-S1 (geo) | ± | ± | ± | ± |
| SASRec-S2 (mixed) | ± | ± | ± | ± |
| *coverage@R* | — | — | — | *상한* |

**그림 1.** 직전→다음 체크인 거리 CDF (R 표시)
**그림 2.** 모델별 E1 vs E2 막대 (부풀림 배수 = E1/E2) + 프로토콜별 모델 순위 변화 (Kendall τ)
**표 3.** 재방문 / 신규 정답 분리 성능 (E4)

---

## 7. 경량 서빙 스펙 (4일차)

### 7.1 구조

```mermaid
flowchart LR
  Q["요청<br/>최근 방문 시퀀스 + 현재 위경도"] --> G["Stage 1-a 지리 필터<br/>BallTree haversine, 반경 R"]
  G --> F["Stage 1-b 임베딩 검색<br/>FAISS IP + IDSelector(반경 내 ID)"]
  F --> C["후보 N=200"]
  C --> R["Stage 2 재랭킹<br/>SASRec 인코더 (ONNX Runtime fp32 / int8)"]
  R --> O["score = h · e_i → Top-10"]
```

- **Stage 1-a**: 현재 위치로 반경 R 내 아이템 ID 집합 G. `|G| < k`면 반경을 2배씩 확장 (동네 범위 넓히기와 같은 제품 동작)
- **Stage 1-b**: Transformer를 돌리지 않는 저렴한 쿼리 `q = 최근 3개 아이템 임베딩 평균`으로 G 안에서 top-N. `|G| ≤ N`이면 생략
  - FAISS `IndexFlatIP` + `IDSelectorBatch`(사전 필터)가 기본. 선택 실험: `IndexHNSWFlat` 전역 검색 후 반경 사후 필터와 recall/latency 비교
- **Stage 2**: SASRec 인코더(시퀀스 → h)만 ONNX로 내보내고, 후보 점수 `h · e_i`는 numpy로 계산
- 솔직하게 적을 점: 아이템 수만 개 규모에선 전체 점수 계산도 충분히 빨라요. 2단계 구조의 의미는 (1) 동네 제약을 **hard filter**로 강제, (2) 아이템 수백만 개로 늘어도 버티는 구조 — 를 보이는 거예요.

### 7.2 API (`dongne_rec/serving/api.py`, FastAPI)

```
POST /recommend
{ "recent_item_ids": [int, ...],   // 최대 50개, 오래된 순
  "lat": float, "lon": float,
  "k": 10, "radius_km": R }
→ { "items": [{ "item_id", "venue_id", "category", "distance_km", "score" }, ...],
    "debug": { "n_geo": int, "n_candidates": int, "radius_used_km": float, "latency_ms": {...} } }
```

### 7.3 측정 방법 (`dongne_rec/serving/bench.py`)

- 요청: test 유저 입력에서 1,000건 이상 재생, warmup 100건, batch=1
- 스레드: 공정 비교용 단일 스레드(`torch.set_num_threads(1)`, ORT `intra_op_num_threads=1`) + 기본 멀티스레드 둘 다
- 단계별(1-a / 1-b / 2)과 end-to-end를 `time.perf_counter_ns`로 재고 p50 / p95 / p99 리포트
- 함께 리포트: **Stage 1 recall@N** (정답이 후보 N개 안에 든 비율) — 2단계 구조의 정확도 손실이 어디서 오는지 보여줌
- HTTP 포함 지연(선택): uvicorn 띄우고 간단한 부하 클라이언트로 p50/p99

### 7.4 최적화 표 (README 표 4)

| 변형 | 모델 크기 (MB) | Stage 2 p50 / p99 (ms) | E2E p50 / p99 (ms) | E4 NDCG@10 | Δ vs fp32 |
|---|---|---|---|---|---|
| PyTorch eager fp32 (CPU) | | | | | — |
| PyTorch MPS | | | | | |
| ONNX Runtime fp32 | | | | | |
| ONNX Runtime int8 (dynamic) | | | | | |
| (선택) 전체 카탈로그 점수 (1단계 없음) | | | | | |

- ONNX 변환: `torch.onnx.export`, opset 17, batch 축 dynamic, 변환 후 PyTorch 대비 최대 절대오차 < 1e-4 확인
- int8: `onnxruntime.quantization.quantize_dynamic` (MatMul 가중치 QInt8)
- 예상되는 정직한 결과: hidden 64 / 2블록은 너무 작아서 int8 이득이 거의 없을 수 있어요. 그러면 hidden 256 모델을 하나 더 만들어 "모델이 커질수록 양자화 이득이 커진다"는 추세를 보여주면 돼요. MPS도 batch=1에서는 디스패치 오버헤드로 CPU보다 느릴 수 있어요 — 이것도 그대로 리포트.

---

## 8. 선택 확장 — 카테고리 텍스트 임베딩으로 cold-start 보강

- `venueCategory`(영문, 예: "Coffee Shop")를 `sentence-transformers/all-MiniLM-L6-v2`(384차원)로 인코딩
- 아이템 임베딩 = `W · text_emb(cat(i))` + 학습형 잔차(0으로 초기화), `W: 384 → 64`
- 평가: 5-core 후라 완전한 cold 아이템은 없으니 **롱테일 구간**(train 빈도 하위 20%)이 정답인 test 케이스에서 E4 NDCG@10을 텍스트 초기화 유무로 비교
- 한계: 카테고리가 거칠어서(같은 카테고리 장소가 많음) "내용 기반 prior" 정도의 역할 — 당근 매물이라면 제목·본문 텍스트로 훨씬 강해질 거라는 문장으로 연결

---

## 9. 레포 구조와 기술 스택

```
dongne-rec/
├── README.md                  # 5일차 산출물 (가설 → 실험 → 결과 → 한계 → A/B)
├── PROJECT_SPEC.md            # 이 문서
├── pyproject.toml
├── Makefile                   # make data / baselines / train / eval / bench
├── configs/
│   ├── data.yaml              # city, k-core, radius_km
│   ├── eval.yaml              # protocols, n_neg=100, ks=[10]
│   └── sasrec.yaml            # 하이퍼파라미터, sampler, geo_ratio, seeds
├── data/                      # .gitignore
│   ├── raw/
│   └── processed/{NYC,TKY}/
├── src/dongne_rec/
│   ├── config.py
│   ├── analysis/  eda.py                                  ← 0일차 ✅
│   ├── data/      preprocess.py ✅, geo.py ✅ (haversine, BallTree, CSR 이웃, GeoIndex), split.py ✅
│   ├── baselines/ pop.py ✅ (MostPop, GeoPop, PersonalFreq), itemknn.py ✅, run.py ✅  ← 1일차
│   ├── models/    sasrec.py
│   ├── train/     samplers.py, trainer.py
│   ├── eval/      protocols.py ✅ (E1~E4 후보 생성·평가), metrics.py ✅, report.py ✅
│   └── serving/   candidates.py, rerank.py, export_onnx.py, quantize.py, api.py, bench.py
├── tests/
│   ├── test_preprocess.py ✅  # 시간 파싱, 연속 중복, k-core 재수렴, ID/좌표/카테고리
│   ├── test_geo.py ✅         # haversine, 이웃 CSR = brute-force
│   ├── test_eda.py ✅         # 전이 쌍, 반경 결정 규칙, end-to-end
│   ├── test_split.py ✅       # 분할 위치, train에 valid/test 정답 없음 (누출 검사)
│   ├── test_metrics.py ✅     # 손계산 예제 + 동점 기댓값 = 무작위 추첨 시뮬레이션
│   ├── test_protocols.py ✅   # negative 제외 규칙, 반경 채우기, 현재 장소 제외, 반경 밖 miss
│   ├── test_baselines.py ✅   # 각 베이스라인 순서, E4에서 GeoPop = MostPop
│   └── test_samplers.py       # geo sampler가 반경 내만 뽑는지
└── results/                   # 실험별 JSONL (config 해시 포함), 표, 그림
```

- 실행 환경: 프로젝트 `.venv` (Python 3.12). `.venv` 안 파일에 macOS hidden 플래그가 붙어 editable 설치의 `.pth`가 무시되는 문제가 있어서, Makefile과 pytest가 `src`를 직접 `PYTHONPATH`에 넣어요. 스크립트를 직접 돌릴 땐 `PYTHONPATH=src .venv/bin/python -m ...`
- Python 3.12, `torch`(MPS), `numpy`, `pandas`, `pyarrow`, `scipy`(희소행렬), `scikit-learn`(BallTree), `faiss-cpu`, `onnx`, `onnxruntime`, `fastapi`, `uvicorn`, `pyyaml`, `matplotlib`, `pytest` / 선택: `sentence-transformers`
- 재현성: 모든 실행에 시드 고정, config를 결과 JSONL에 같이 저장, `make all`로 처음부터 표까지 재생성
- MPS에서 미지원 연산이 나오면 `PYTORCH_ENABLE_MPS_FALLBACK=1`, 모델이 작아서 CPU가 더 빠를 수도 있으니 첫 학습 때 둘 다 재보기

---

## 10. 일정 (날짜 확정판)

| 날짜 | 단계 | 할 일 | 완료 기준 (DoD) |
|---|---|---|---|
| **10/31(토)** | 0일차 준비 ✅ | 환경 세팅, 데이터 다운로드, 전처리, EDA | 표 1 초안, **R = 3km 확정**, 거리 CDF 그림 |
| **11/1(일)** | 1일차 ✅ (10/31에 앞당겨 완료) | 분할, E1~E4 평가 하네스, 베이스라인 4종, 분할·지표 테스트 | 표 2의 베이스라인 행 채움, `pytest` 통과 (25개) |
| **11/2(월)** | 2일차 | SASRec 구현, S0 학습, 베이스라인 대비 sanity check | S0이 E3에서 MostPop보다 높음 (아니면 버그부터 의심), PersonalFreq와의 차이와 신규 정답 구간 확인 |
| **11/3(화)** | 3일차 | S1/S2 sampler, 9회 학습, 표 2·3, 그림 2, bootstrap | 표 2 완성 → **MVP 컷라인** |
| **11/4(수)** | 4일차 | ONNX 변환, int8, 2단계 파이프라인, FastAPI, 벤치 | 표 4 완성, Stage 1 recall@N |
| **11/5(목)** | 5일차 | README 작성, A/B 단락, 그림 정리, 레포 공개 | 처음 보는 사람이 README만 읽고 3가지 주장을 이해 |
| **11/6(금)** | 제출 | 최종 검토 후 지원 | — |

**MVP 컷라인**: 11/5 저녁까지 서빙이 덜 끝나면 1~3일차 결과(표 1·2·3 + 그림 1·2)로 README를 마무리해서 먼저 제출하고, 서빙은 인터뷰 전까지 보강해요 (원 플랜과 동일).

---

## 11. 리스크와 대응

| 리스크 | 신호 | 대응 |
|---|---|---|
| SASRec이 ItemKNN/GeoPop을 못 이김 | 2일차 표 | 데이터 규모 한계일 수 있음. 숨기지 말고 "작은 데이터에선 단순 모델이 강하다"를 발견으로 리포트, dropout↑·hidden↓ 시도 |
| SASRec이 PersonalFreq(재방문)를 못 넘음 | 2일차 표 (1일차 기준 E4 0.239) | 재방문 / 신규 구간을 분리해서 SASRec이 신규 구간에서 이기는지로 판단. 시간 되면 "재방문은 PersonalFreq, 신규는 SASRec" 혼합을 대조군으로 |
| H1 효과가 없거나 반대 | 3일차 표 | false negative 해석 + S2 mixed 결과로 trade-off 서술. 가설이 기각돼도 근거가 탄탄하면 좋은 결과 |
| 5-core 후 데이터가 너무 작음 | 0일차 | 아이템 ≥ 3으로 완화, 또는 Tokyo로 교체 |
| Manhattan에서 반경 내 장소가 너무 많아 메모리 부족 | 0일차 이웃 수 히스토그램 | CSR int32 사용, 그래도 크면 R 축소 또는 격자 셀 근사 |
| MPS 미지원 연산 / 느림 | 2일차 첫 학습 | fallback env 설정, CPU 학습으로 전환 (수 분 규모라 문제없음) |
| ONNX 변환 실패 | 4일차 | 어텐션 직접 구현으로 사전 회피, 그래도 막히면 TorchScript로 대체하고 표에 명시 |
| 시간 초과 | 매일 저녁 DoD 체크 | MVP 컷라인 적용, 선택 확장은 제출 후로 |

---

## 12. README 골격 (5일차)

1. **TL;DR** — 3가지 주장, 숫자 포함, 3줄
2. **문제** — 동네 제약이 왜 추천을 어렵게 만드는가 (당근의 문제의식과 연결)
3. **데이터** — 표 1, 거리 CDF, R 선택 근거
4. **가설** — H1, H2를 각각 한 문장 + 예상 메커니즘
5. **실험 설정** — 분할, E1~E4, S0~S2, 시드/통계
6. **결과** — 표 2·3, 그림 2, 핵심 해석 3~4개
7. **서빙** — 구조도, 표 4, Stage 1 recall
8. **한계** — LOO 시간 누출, 체크인 ≠ 거래, 데이터 시기(2012~13)·규모, 위치 = 직전 장소라는 근사, false negative
9. **실서비스라면: 온라인 A/B 설계** (아래 초안)
10. 재현 방법 — `make all`

### A/B 테스트 단락 초안 (공고가 온라인 실험 검증을 강조하므로 비중 있게)

> 오프라인 E4 개선이 실제 가치로 이어지는지는 온라인 실험으로만 확인할 수 있어요. 실서비스라면 다음과 같이 설계하겠습니다.
> - **가설**: 동네 내 negative로 학습한 모델이 동네 추천 지면의 관련성을 높여 전환을 늘린다.
> - **1차 지표**: 추천 지면 노출 대비 클릭 → 채팅 시작(거래 의도) 전환율 (유저 단위 집계). POI 맥락이라면 방문 전환.
> - **보조 지표**: 세션당 탐색 깊이, 7일 재방문.
> - **가드레일**: p99 latency, 신규 매물(cold-start) 노출 비중, 판매자 측 노출 집중도(Gini), 숨김/신고율.
> - **랜덤화 단위**: 기본은 유저 단위. 다만 같은 동네의 매물과 구매자가 서로 경쟁하는 양면 마켓이라 처치군의 행동이 대조군 노출에 영향을 줄 수 있어(간섭), 효과가 크면 지역(동네 클러스터) 단위 랜덤화로 교차 검증.
> - **기간·검정력**: 요일 효과를 덮도록 최소 2주, 사전에 MDE와 표본 수 계산, 초기 novelty 효과는 첫 주를 따로 분석.
> - **오프라인-온라인 정합성**: E1이 아닌 E4가 온라인 지표와 같은 방향으로 움직이는지 확인하는 것 자체가 이 프로젝트 H2의 실전 검증.

---

## 13. 인터뷰 대비 포인트

- **왜 sampled metric이 위험한가?** → E1 vs E2/E3 결과로 답하기, 순위 뒤집힘 사례
- **geo negative의 false negative는?** → S2 mixed와 재방문/신규 분리 결과로 답하기
- **동네 제약을 hard filter로 넣을지 모델 feature로 넣을지?** → 이 프로젝트는 hard filter(서빙) + 학습 분포 조정(sampler). feature 방식(거리 임베딩, 위치 인코딩)은 후속 과제
- **2단계에서 잃는 정확도는?** → Stage 1 recall@N과 E2E vs E4 차이
- **int8이 왜 (안) 빨라졌나?** → 모델 크기, 연산 비중(임베딩 조회 vs MatMul), 스레드 설정
- **당근 실데이터라면 뭐가 다른가?** → 매물은 팔리면 사라짐(아이템 수명 짧음 → cold-start 상시), 판매자-구매자 양면성, 동네 인증으로 위치 신뢰도가 높음, 텍스트·이미지 등 콘텐츠 신호가 풍부함
- **당근의 SIGIR 2026 논문과의 연결** → "이용자가 경험할 수 있는 범위 안의 학습 데이터"라는 문제의식을 공개 데이터의 negative 샘플링 분포 문제로 작게 재현한 것 (논문 세부 내용은 지원 전에 직접 읽고 정확히 인용)

---

## 14. 출처

- 당근 채용 공고 – Software Engineer, Machine Learning
- 바이라인네트워크, "당근의 추천시스템은 LLM 안 부럽다"
- Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020
- Yang et al., *Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs*, IEEE TSMC 2015 (데이터셋)
