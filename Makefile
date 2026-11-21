.PHONY: help setup test test-e2e benchmark gui cli arena report clean

PYTHON := .venv/bin/python3
PYTEST := .venv/bin/pytest

help:
	@echo "Available commands for Chess AI:"
	@echo "  make setup       Install virtualenv and all dependencies"
	@echo "  make test        Run complete 129-test automated pyramid"
	@echo "  make test-e2e    Run Playwright headless Chromium E2E tests"
	@echo "  make benchmark   Run Maia ONNX vs Stockfish & ACL Governor benchmarks"
	@echo "  make gui         Launch PyQt6 Master Graphical Dashboard"
	@echo "  make cli         Run full 15-phase CLI diagnostic audit"
	@echo "  make arena       Run multi-game autonomous AI arena simulation"
	@echo "  make report      Generate standalone interactive HTML visual report"
	@echo "  make clean       Remove temporary caches and build artifacts"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/playwright install chromium
	chmod +x bin/stockfish 2>/dev/null || true

test:
	$(PYTEST) tests/ -v

test-e2e:
	$(PYTEST) tests/test_playwright_e2e.py tests/test_e2e_playwright_extension.py -v

benchmark:
	$(PYTHON) scripts/benchmark_maia_and_acl.py

gui:
	$(PYTHON) main.py --gui

cli:
	$(PYTHON) main.py --cli

arena:
	$(PYTHON) main.py --arena --games 2 --white god --black 1500

report:
	$(PYTHON) generate_visual_report.py
	@open web_preview.html 2>/dev/null || xdg-open web_preview.html 2>/dev/null || echo "Report generated at web_preview.html"

clean:
	rm -rf .pytest_cache/ htmlcov/ .coverage* build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
