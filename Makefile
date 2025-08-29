PYTHON := python
EXPORT_ENV := export PYTHONPATH=.

serve:
	$(EXPORT_ENV); uvicorn app.api.main:app --reload

test:
	pytest -q

artifacts:
	curl -s http://127.0.0.1:8000/artifacts | $(PYTHON) -m json.tool | sed -n '1,120p'

ensemble:
	curl -s http://127.0.0.1:8000/forecast/ensemble | $(PYTHON) -m json.tool

regional:
	curl -s "http://127.0.0.1:8000/forecast/regional?region=americas" | $(PYTHON) -m json.tool

log-tail:
	tail -n 20 data/prediction_log.csv || true

tidy:
	$(PYTHON) scripts/backtest_from_log.py

backtest:
	$(PYTHON) scripts/backtest_join_actuals.py data/spx_actuals.csv

art-metrics:
	$(PYTHON) scripts/generate_dummy_metrics.py
