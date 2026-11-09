VENV ?= .venv
PY ?= $(VENV)/bin/python
ifeq ($(shell uname -sm),Darwin arm64)
LOCK ?= requirements-macos-arm64-py312.txt
endif
PYTHON ?= $(if $(strip $(LOCK)),python3.12,python3)
CITY ?= NYC
CONFIG ?= configs/data.yaml
EVAL_CONFIG ?= configs/eval.yaml
SASREC_CONFIG ?= configs/sasrec.yaml
SAMPLER ?= random
SEED ?= 0
R ?=
SET ?=
TAG ?=
RUNS ?=
RUN ?=
# .venv 안 파일에 macOS hidden 플래그가 붙으면 Python이 editable 설치의 .pth를 무시해서 src를 직접 넣어요.
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: setup data eda geo baselines train compare export bench http-bench serve test

setup:
	$(PYTHON) -m venv "$(VENV)"
ifneq ($(strip $(LOCK)),)
	$(PY) -m pip install --only-binary=:all: --no-deps -r "$(LOCK)"
	$(PY) -m pip install --no-deps --no-build-isolation -e ".[dev]"
else
	$(PY) -m pip install -e ".[dev]"
endif
	$(PY) -m pip check

# 원본 CSV → data/processed/$(CITY)/{interactions,items,users}.parquet + stats.json
data:
	$(PY) -m dongne_rec.data.preprocess --config $(CONFIG) --city $(CITY)

# → results/eda/$(CITY)/ (eda_report.md, eda_stats.json, 그림)
eda:
	$(PY) -m dongne_rec.analysis.eda --config $(CONFIG) --city $(CITY)

# 반경·좌표에 따라 이웃 CSR을 캐시합니다.
geo:
	$(PY) -m dongne_rec.data.geo --config $(CONFIG) --city $(CITY) $(if $(R),--radius-km $(R))

# → results/runs/$(CITY)/R{r}/{run_id}/
baselines:
	$(PY) -m dongne_rec.baselines.run --config $(CONFIG) --eval-config $(EVAL_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R)) $(if $(TAG),--tag "$(TAG)")

# SASRec 1회 학습 → results/runs/$(CITY)/R{r}/{run_id}/
train:
	$(PY) -m dongne_rec.train.trainer --config $(CONFIG) --eval-config $(EVAL_CONFIG) --sasrec-config $(SASREC_CONFIG) --city $(CITY) --sampler $(SAMPLER) --seed $(SEED) $(if $(R),--radius-km $(R)) $(if $(TAG),--tag "$(TAG)") $(foreach value,$(SET),--set "$(value)")

# 비교할 실행을 RUNS로 지정할 수 있습니다. → results/reports/$(CITY)/R{r}/{report_id}/
compare:
	$(PY) -m dongne_rec.eval.compare --config $(CONFIG) --eval-config $(EVAL_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R)) $(foreach run,$(RUNS),--run "$(run)")

# 서빙 (configs/serving.yaml의 model.sampler/seed 체크포인트)
SERVING_CONFIG ?= configs/serving.yaml
ARTIFACTS ?=
BACKEND ?= onnx-int8

# → artifacts/$(CITY)/R{r}/{sampler}_s{seed}/ (ONNX fp32·int8, 아이템 임베딩, 장소 메타)
export:
	$(PY) -m dongne_rec.serving.export_onnx --config $(CONFIG) --eval-config $(EVAL_CONFIG) --serving-config $(SERVING_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R)) $(if $(RUN),--run "$(RUN)")

# → results/serving/$(CITY)/R{r}/{sampler}_s{seed}/{bench_id}/bench.md
bench:
	$(PY) -m dongne_rec.serving.bench --config $(CONFIG) --eval-config $(EVAL_CONFIG) --serving-config $(SERVING_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R))

# → results/serving/$(CITY)/R{r}/{sampler}_s{seed}/{http_id}/http.md
http-bench:
	$(PY) -m dongne_rec.serving.http_bench --config $(CONFIG) --eval-config $(EVAL_CONFIG) --serving-config $(SERVING_CONFIG) --city $(CITY) $(if $(R),--radius-km $(R))

serve:
	DONGNE_SERVING_CONFIG="$(SERVING_CONFIG)" DONGNE_DATA_CONFIG="$(CONFIG)" DONGNE_CITY="$(CITY)" DONGNE_RADIUS_KM="$(R)" \
	DONGNE_ARTIFACTS="$(ARTIFACTS)" DONGNE_BACKEND="$(BACKEND)" $(PY) -m uvicorn dongne_rec.serving.api:create_app --factory --port 8000

test:
	$(PY) -m pytest -q
