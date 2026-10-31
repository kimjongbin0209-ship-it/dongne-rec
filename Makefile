PY ?= .venv/bin/python
CITY ?= NYC
CONFIG ?= configs/data.yaml
EVAL_CONFIG ?= configs/eval.yaml
SASREC_CONFIG ?= configs/sasrec.yaml
SAMPLER ?= random
SEED ?= 0
R ?=
# .venv 안 파일에 macOS hidden 플래그가 붙으면 Python이 editable 설치의 .pth를 무시해서 src를 직접 넣어요.
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: setup data eda geo baselines train compare export bench serve test

setup:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

# 원본 CSV → data/processed/$(CITY)/{interactions,items,users}.parquet + stats.json
data:
	$(PY) -m dongne_rec.data.preprocess --config $(CONFIG) --city $(CITY)

# → results/eda/$(CITY)/ (eda_report.md, eda_stats.json, 그림)
eda:
	$(PY) -m dongne_rec.analysis.eda --config $(CONFIG) --city $(CITY)

# 반경 이웃 CSR. R을 안 주면 config의 geo.radius_km 사용 (예: make geo R=3)
geo:
	$(PY) -m dongne_rec.data.geo --config $(CONFIG) --city $(CITY) $(if $(R),--radius-km $(R))

# → results/baselines/$(CITY)/R{r}/ (summary.md, summary.json, per_query.parquet). R을 주면 민감도 분석 (예: make baselines R=2)
baselines:
	$(PY) -m dongne_rec.baselines.run --config $(CONFIG) --eval-config $(EVAL_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R))

# SASRec 1회 학습 → results/sasrec/$(CITY)/R{r}/$(SAMPLER)_s$(SEED)/ (예: make train SAMPLER=random SEED=1)
train:
	$(PY) -m dongne_rec.train.trainer --config $(CONFIG) --eval-config $(EVAL_CONFIG) --sasrec-config $(SASREC_CONFIG) --city $(CITY) --sampler $(SAMPLER) --seed $(SEED)

# 베이스라인 + SASRec 전체 비교 → results/summary/$(CITY)/R{r}/table2.md
compare:
	$(PY) -m dongne_rec.eval.compare --config $(CONFIG) --eval-config $(EVAL_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R))

# 서빙 (configs/serving.yaml의 model.sampler/seed 체크포인트)
SERVING_CONFIG ?= configs/serving.yaml
ARTIFACTS ?= artifacts/$(CITY)/R3/mixed_s0
BACKEND ?= onnx-int8

# → artifacts/$(CITY)/R{r}/{sampler}_s{seed}/ (ONNX fp32·int8, 아이템 임베딩, 장소 메타)
export:
	$(PY) -m dongne_rec.serving.export_onnx --config $(CONFIG) --eval-config $(EVAL_CONFIG) --serving-config $(SERVING_CONFIG) --city $(CITY)

# → results/serving/$(CITY)/R{r}/{sampler}_s{seed}/bench.md
bench:
	$(PY) -m dongne_rec.serving.bench --config $(CONFIG) --eval-config $(EVAL_CONFIG) --serving-config $(SERVING_CONFIG) --city $(CITY)

serve:
	DONGNE_ARTIFACTS=$(ARTIFACTS) DONGNE_BACKEND=$(BACKEND) $(PY) -m uvicorn dongne_rec.serving.api:create_app --factory --port 8000

test:
	$(PY) -m pytest -q
