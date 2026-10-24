# ==============================================================================
# OmniIntel Automation Makefile
# Unified CLI commands for local development, testing, and maintenance.
# ==============================================================================

.PHONY: help setup test lint clean

# Default shell
SHELL := /usr/bin/env bash
PYTHON ?= $(shell if [ -x ./.venv/bin/python3 ]; then echo ./.venv/bin/python3; elif [ -x /opt/homebrew/bin/python3 ]; then echo /opt/homebrew/bin/python3; elif [ -x /usr/bin/python3 ]; then echo /usr/bin/python3; else echo python3; fi)

help:
	@echo "OmniIntel Project Automation Commands:"
	@echo "  make setup    - Validate environment, initialize .venv, and setup .env"
	@echo "  make test     - Run full automated unit and integration test suite"
	@echo "  make lint     - Verify Python syntax and AST integrity across codebase"
	@echo "  make clean    - Remove build artifacts, pycache, and temporary test caches"

setup:
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh

test:
	@echo "Running test suite via Python unittest..."
	@$(PYTHON) -m unittest discover tests -v

lint:
	@echo "Checking Python syntax across src/ and tests/..."
	@$(PYTHON) -m compileall -q src tests
	@echo "✓ Syntax validation passed cleanly."

clean:
	@echo "Cleaning temporary bytecode and test artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✓ Workspace clean."
