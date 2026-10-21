.PHONY: format lint structure test quality

format:
	.venv/bin/ruff format app tests scripts

lint:
	.venv/bin/ruff check app tests scripts

structure:
	.venv/bin/python scripts/check_structure.py

test:
	.venv/bin/python -m pytest -q

quality: lint structure test
	.venv/bin/ruff format --check app tests scripts
