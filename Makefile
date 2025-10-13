# Makefile for FRA Diagnostics Platform
# SIH 2025 PS 25190

.PHONY: help install install-dev test test-coverage lint format security clean run deploy docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit
SAFETY := $(PYTHON) -m safety
STREAMLIT := streamlit

help: ## Show this help message
	@echo "FRA Diagnostics Platform - Available Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# INSTALLATION
# ============================================================================

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Production dependencies installed"

install-dev: install ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "✓ Development dependencies installed"
	@echo "✓ Pre-commit hooks installed"

install-all: install-dev ## Install all dependencies (production + development)
	@echo "✓ All dependencies installed"

# ============================================================================
# TESTING
# ============================================================================

test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-fast: ## Run tests without slow tests
	$(PYTEST) tests/ -v -m "not slow"

test-coverage: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "✓ Coverage report generated in htmlcov/index.html"

test-verbose: ## Run tests with verbose output
	$(PYTEST) tests/ -vv --tb=long

test-parallel: ## Run tests in parallel
	$(PYTEST) tests/ -n auto

test-watch: ## Run tests in watch mode
	$(PYTEST) tests/ --watch

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: ## Run all linters
	@echo "Running flake8..."
	$(FLAKE8) . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(FLAKE8) . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	@echo "Running pylint..."
	-pylint *.py
	@echo "✓ Linting complete"

format: ## Format code with black and isort
	@echo "Formatting with black..."
	$(BLACK) . --line-length=100
	@echo "Sorting imports with isort..."
	$(ISORT) . --profile=black --line-length=100
	@echo "✓ Code formatted"

format-check: ## Check if code is formatted correctly
	$(BLACK) . --check --line-length=100
	$(ISORT) . --check-only --profile=black --line-length=100

type-check: ## Run type checking with mypy
	$(MYPY) . --ignore-missing-imports
	@echo "✓ Type checking complete"

# ============================================================================
# SECURITY
# ============================================================================

security: ## Run security checks
	@echo "Running bandit security scanner..."
	$(BANDIT) -r . -ll --skip B101,B601
	@echo "Checking for vulnerable dependencies..."
	$(SAFETY) check --json
	pip-audit
	@echo "✓ Security checks complete"

security-report: ## Generate detailed security report
	$(BANDIT) -r . -f html -o security-report.html
	@echo "✓ Security report generated: security-report.html"

# ============================================================================
# CLEANUP
# ============================================================================

clean: ## Remove build artifacts and cache
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name 'htmlcov' -exec rm -rf {} +
	find . -type f -name '.coverage' -delete
	rm -rf build/ dist/ temp/*.* logs/*.log
	@echo "✓ Cleanup complete"

clean-all: clean ## Remove all generated files including models
	rm -rf models/*.pth models/*.onnx models/*.pkl
	rm -rf data/processed/* synthetic_data/*
	@echo "✓ Deep cleanup complete"

# ============================================================================
# APPLICATION
# ============================================================================

run: ## Run Streamlit application
	$(STREAMLIT) run app.py

run-debug: ## Run application in debug mode
	ENVIRONMENT=development LOG_LEVEL=DEBUG $(STREAMLIT) run app.py

run-prod: ## Run application in production mode
	ENVIRONMENT=production $(STREAMLIT) run app.py --server.port=8501 --server.address=0.0.0.0

# ============================================================================
# DEPLOYMENT
# ============================================================================

deploy-pi: ## Deploy to Raspberry Pi
	chmod +x pi_deploy.sh
	./pi_deploy.sh

build: ## Build distribution packages
	$(PYTHON) -m build
	@echo "✓ Distribution packages built in dist/"

# ============================================================================
# DATA & MODELS
# ============================================================================

generate-data: ## Generate synthetic FRA data
	$(PYTHON) simulator.py

train-models: ## Train AI ensemble models
	$(PYTHON) ai_ensemble.py

export-onnx: ## Export models to ONNX format
	$(PYTHON) onnx_export.py

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs: ## Generate documentation
	cd docs && make html
	@echo "✓ Documentation generated in docs/_build/html/index.html"

docs-serve: docs ## Serve documentation locally
	$(PYTHON) -m http.server --directory docs/_build/html 8000

# ============================================================================
# DEVELOPMENT WORKFLOW
# ============================================================================

pre-commit: format lint test-fast ## Run pre-commit checks
	@echo "✓ Pre-commit checks passed"

ci: format-check lint type-check security test-coverage ## Run all CI checks
	@echo "✓ All CI checks passed"

check: format lint type-check security ## Run all quality checks without tests
	@echo "✓ All quality checks passed"

init: ## Initialize development environment
	@echo "Initializing development environment..."
	cp .env.example .env
	$(MAKE) install-dev
	@echo "✓ Development environment initialized"
	@echo "✓ Please edit .env file with your configuration"

# ============================================================================
# MONITORING
# ============================================================================

profile: ## Profile application performance
	$(PYTHON) -m cProfile -o profile.stats app.py
	$(PYTHON) -m pstats profile.stats

memory-profile: ## Profile memory usage
	$(PYTHON) -m memory_profiler app.py

# ============================================================================
# UTILITY
# ============================================================================

freeze: ## Freeze current dependencies
	$(PIP) freeze > requirements.lock
	@echo "✓ Dependencies frozen to requirements.lock"

upgrade: ## Upgrade all dependencies
	$(PIP) install --upgrade -r requirements.txt -r requirements-dev.txt
	@echo "✓ Dependencies upgraded"

info: ## Show project information
	@echo "FRA Diagnostics Platform"
	@echo "========================"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Project directory: $$(pwd)"
	@echo "Environment: $${ENVIRONMENT:-not set}"
