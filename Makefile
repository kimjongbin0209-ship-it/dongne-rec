PY ?= .venv/bin/python
CITY ?= NYC
CONFIG ?= configs/data.yaml
EVAL_CONFIG ?= configs/eval.yaml
R ?=
# .venv 안 파일에 macOS hidden 플래그가 붙으면 Python이 editable 설치의 .pth를 무시해서 src를 직접 넣어요.
export PYTHONPATH := src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: setup data eda geo baselines test

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

test:
	$(PY) -m pytest -q
