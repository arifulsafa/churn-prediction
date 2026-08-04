.PHONY: install data eda train test api diagram all clean

install:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

data:
	.venv/bin/python -m src.data_generation

eda:
	.venv/bin/python -m src.eda

train:
	.venv/bin/python -m src.train

test:
	.venv/bin/python -m pytest

api:
	.venv/bin/uvicorn src.api:app --reload --port 8000

diagram:
	.venv/bin/python docs/make_architecture_diagram.py

all: data eda train test

clean:
	rm -rf artifacts/*.joblib artifacts/*.json reports/* __pycache__ .pytest_cache
