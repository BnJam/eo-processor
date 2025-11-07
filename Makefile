# ==============================================================================
# Project Variables
# ==============================================================================
# The default virtual environment directory for uv
VENV_DIR = .venv
# Use 'uv run' for all commands to automatically activate the environment
PYTHON_RUN = uv run python
PYTEST_RUN = uv run pytest

# ==============================================================================
# Default and Setup Targets
# ==============================================================================

.PHONY: all setup sync develop build clean test lint help

all: develop

setup: ## Create the virtual environment and sync dependencies
	@echo "🛠️ Creating virtual environment and syncing dependencies with uv..."
	uv venv  # Creates the .venv directory if it doesn't exist
	uv sync  # Installs project and dev dependencies from pyproject.toml

sync: ## Sync dependencies (install/update packages)
	@echo "🔄 Syncing dependencies with uv..."
	uv sync

develop: sync ## Install the Rust code as a Python module for development
	@echo "🔨 Installing native extension in development mode..."
	# maturin uses its own logic for development install, and it's best to run
	# it inside the uv-managed environment using 'uv run'.
	# The '--uv' flag tells maturin to use uv for installation steps.
	uv run maturin develop --uv

# ==============================================================================
# Build, Clean, and Utility Targets
# ==============================================================================

build: sync ## Build the release wheels for distribution
	@echo "⚙️ Building release wheels..."
	# The build command uses the project's requirements via PEP 517
	uv build --release --out dist

install: build ## Install the project from the built wheel
	@echo "📦 Installing built wheel into environment..."
	# Find the latest built wheel and install it
	$(PYTHON_RUN) -m pip install --force-reinstall dist/*.whl

clean: ## Clean up build artifacts
	@echo "🧹 Cleaning up..."
	# Remove Rust/Cargo build artifacts
	cargo clean
	# Remove Python-related build directories
	rm -rf dist target/wheels build *.egg-info
	# Remove the native extension file created by 'maturin develop'
	find . -type f -name '*.so' -delete || true
	# Remove the virtual environment
	rm -rf $(VENV_DIR)

test: develop ## Run tests (assuming you use pytest)
	@echo "🧪 Running tests..."
	$(PYTEST_RUN)

lint: ## Run linters (customize with your preferred uv-managed tools)
	@echo "🔍 Running linters..."
	# Example: uv run ruff check .
	# Example: uv run black . --check
	# Add your actual lint commands here

# ==============================================================================
# Help Target
# ==============================================================================

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
