.PHONY: setup test lint env audit
setup:
	uv sync --extra dev
test:
	uv run pytest -q
lint:
	uv run ruff check src tests scripts
env:
	uv run python scripts/env_report.py
audit:
	uv run python scripts/audit_dataset.py --data-dir data/raw --output-dir results/reports
