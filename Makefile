# ==============================================================================
# Project Variables
# ==============================================================================

# The default virtual environment directory for uv
VENV_DIR = .venv

# Use direct commands, assuming the virtual environment is manually activated
PYTHON_RUN = python
PYTEST_RUN = $(VENV_DIR)/bin/pytest

# ==============================================================================
# Default and Setup Targets
# ==============================================================================

.PHONY: all setup sync develop build install clean test lint docs help

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
	# NOTE: This target assumes the virtual environment is manually activated (e.g., source .venv/bin/activate)
	maturin develop

# ==============================================================================
# Build, Clean, and Utility Targets
# ==============================================================================

build: sync ## Build the release wheels for distribution
	@echo "⚙️ Building release wheels..."
	# NOTE: This target assumes the virtual environment is manually activated
	uv run maturin build --release --out dist

install: build ## Install the project from the built wheel
	@echo "📦 Installing built wheel into environment..."
	# Find the latest built wheel and install it
	uv pip uninstall eo_processor || true
	uv pip install .[dask]

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

test: ## Run tests with tox
	@echo "🧪 Running tests..."
	tox


lint: ## Run linters (customize with your preferred uv-managed tools)
	@echo "🔍 Running linters..."
	tox -e lint

docs: ## Build Sphinx HTML documentation (outputs to docs/build/html)
	@echo "📘 Building Sphinx documentation..."
	uv run sphinx-build -b html docs/source docs/build/html

# ==============================================================================
# Help Target
# ==============================================================================

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@grep -E '^$$a-zA-Z\_-$$+:.?## .$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------
# Documentation Extended Targets
# ------------------------------------------------------------------------------

# Remove previously built Sphinx HTML to ensure a clean rebuild
docs-clean: ## Remove built documentation (docs/build)
	@echo "🧹 Cleaning built documentation..."
	rm -rf docs/build

# Build docs after cleaning (explicit full rebuild)
docs-rebuild: docs-clean docs ## Clean and rebuild documentation

# Open the local documentation index (macOS 'open', fallback to xdg-open)
docs-open: ## Open docs in default browser (build first if missing)
	@if [ ! -d docs/build/html ]; then \
        echo "⚠️  Docs build directory missing. Building now..."; \
        $(MAKE) docs; \
    fi
	@if [ ! -f docs/build/html/index.html ]; then \
		echo "⚠️  Docs not built yet. Building now..."; \
		$(MAKE) docs; \
	fi
	@echo "🌐 Opening documentation..."
	@if command -v open >/dev/null 2>&1; then open docs/build/html/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then xdg-open docs/build/html/index.html; \
	else echo "Please open docs/build/html/index.html manually."; fi
