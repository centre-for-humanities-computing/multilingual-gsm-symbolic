add-uv:
	@echo "--- 🚀 Installing UV ---"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	# windows:
	# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

install:
	@echo "--- 🚀 Installing project ---"
	uv sync

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format .                            # running ruff formatting
	uv run ruff check . --fix                       # running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	uv run ruff format . --check                    # running ruff formatting
	uv run ruff check .                             # running ruff linting

type-check:
	@echo "--- 🔍 Running type checker ---"
	uv run ty check src/multilingual_gsm_symbolic

test:
	@echo "--- 🧪 Running tests ---"
	uv run pytest -vv
