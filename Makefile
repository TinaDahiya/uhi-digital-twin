.PHONY: install test preflight export fetch preprocess
install:
	pip install -e ".[gee,geo,parquet,notebook,dev]"
test:
	pytest -q
preflight:
	python scripts/01_export_gee.py --preflight-only
export:
	python scripts/01_export_gee.py
fetch:
	python scripts/02_fetch_gee.py
preprocess:
	python scripts/03_preprocess.py
