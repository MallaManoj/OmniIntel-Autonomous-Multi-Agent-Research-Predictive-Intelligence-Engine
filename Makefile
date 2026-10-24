# ==============================================================================
# OmniIntel Automation Makefile
# Unified CLI commands for local development, testing, and maintenance.
# ==============================================================================

.PHONY: help setup test lint clean train benchmark demo-rag demo-agents run-api docker-build docker-run compose-up compose-down compose-logs

# Default shell
SHELL := /usr/bin/env bash
PYTHON ?= $(shell if [ -x ./.venv/bin/python3 ]; then echo ./.venv/bin/python3; elif [ -x /opt/homebrew/bin/python3 ]; then echo /opt/homebrew/bin/python3; elif [ -x /usr/bin/python3 ]; then echo /usr/bin/python3; else echo python3; fi)

help:
	@echo "OmniIntel Project Automation Commands:"
	@echo "  make setup       - Validate environment, initialize .venv, and setup .env"
	@echo "  make test        - Run full automated unit and integration test suite"
	@echo "  make lint        - Verify Python syntax and AST integrity across codebase"
	@echo "  make train       - Execute PyTorch Cross-Encoder training and validation loop"
	@echo "  make benchmark   - Compare PyTorch Cross-Encoder vs. Vector Baseline"
	@echo "  make demo-rag    - Execute live end-to-end Knowledge Graph & Hybrid RAG demonstration"
	@echo "  make demo-agents - Execute live Autonomous Multi-Agent & Dynamic Self-Correction demo"
	@echo "  make run-api     - Launch FastAPI backend server with auto-reload (port 8000)"
	@echo "  make clean       - Remove build artifacts, pycache, and temporary test caches"

setup:
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh

train:
	@echo "Launching PyTorch Cross-Encoder fine-tuning pipeline..."
	@$(PYTHON) scripts/train_reranker.py

benchmark:
	@echo "Running quantitative PyTorch Cross-Encoder vs Vector Search benchmarks..."
	@$(PYTHON) scripts/benchmark_models.py

demo-rag:
	@echo "Executing live Knowledge Graph & Hybrid RAG demonstration..."
	@$(PYTHON) scripts/demo_rag_pipeline.py

demo-agents:
	@echo "Executing live OmniIntel Autonomous Multi-Agent Interactive Demonstration..."
	@$(PYTHON) scripts/demo_multi_agent_system.py

run-api:
	@echo "Starting OmniIntel FastAPI production service on http://0.0.0.0:8000..."
	@$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	@echo "Building production multi-stage Docker image (omniintel:latest)..."
	@docker build -t omniintel:latest .

docker-run:
	@echo "Running OmniIntel container on http://localhost:8000..."
	@docker run --rm -p 8000:8000 --name omniintel-service omniintel:latest

compose-up:
	@echo "Spinning up OmniIntel multi-service cluster (API + UI + Data Volumes)..."
	@docker compose up --build -d

compose-down:
	@echo "Stopping OmniIntel cluster and dismantling network..."
	@docker compose down

compose-logs:
	@echo "Tailing multi-service cluster logs..."
	@docker compose logs -f

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
