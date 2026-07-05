PYTHON ?= python

.PHONY: test_all submit_all check install clean lint format typecheck

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  submit_*    Run \`submit.py\` for a puzzle where * is the index of"
	@echo "                 the puzzle (e.g. submit_a)"
	@echo "  submit_all  Run \`submit.py\` for all puzzles"
	@echo "  test_*      Run the test suite for a puzzle where * is the index"
	@echo "                 of the puzzle (e.g. test_a)"
	@echo "  test_all    Run the test suite"
	@echo ""
	@echo "  install    Install the environment (runs prerequisite checks first)"
	@echo "  clean      Remove the virtual environment"
	@echo "  lint       Check code with ruff"
	@echo "  format     Format code with ruff"
	@echo "  typecheck  Type-check code with mypy"
	@echo "  lab        Launch JupyterLab"
	@echo ""


test_all:
	pytest tests/

test_%:
	pytest tests/test_$*_*.py

submit_all:
	$(PYTHON) submit.py --all

submit_%:
	$(PYTHON) submit.py $*

check:
	@which git > /dev/null 2>&1 || (echo "Error: git is not installed"; exit 1)
	@which uv  > /dev/null 2>&1 || (echo "Error: uv is not installed";  exit 1)

install: check
	uv sync
	uv run pre-commit install

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy .

activate:
	source .venv/bin/activate
