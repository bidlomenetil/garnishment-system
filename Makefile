.PHONY: install run test live check lint typecheck eval

UV := $(shell command -v uv 2>/dev/null)

install:
ifndef UV
	@echo "uv is not installed. Install it with:"
	@echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
	@exit 1
else
	uv sync --all-groups
endif

run:
	uv run uvicorn garnishment.api.app:create_app --factory --host 0.0.0.0 --port 8000

test:
	uv run pytest -m "not live" -q

# Needs LLM_PROVIDER set to a real provider and its API key; skips without one.
live:
	uv run pytest -m live -q

check: lint typecheck test

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

eval:
	uv run python evals/run_eval.py
