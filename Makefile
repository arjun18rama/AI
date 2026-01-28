.PHONY: install install-dev train lint format test

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt

train:
	python train.py --config configs/default.yaml

lint:
	ruff check .

format:
	black .

test:
	pytest
