# Diabetes Prediction ML Pipeline - Makefile

.PHONY: help setup install clean run-all predict check-config show-results

help:  ## Show this help message
	@echo "🧠 Diabetes Prediction ML Pipeline"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $1, $2}'

setup:  ## Setup project directories and environment
	python setup.py

install:  ## Install Python dependencies
	pip install -r requirements.txt

check-config:  ## Check configuration and system readiness
	python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.') / 'src')); from src.core.config_loader import config_loader; print('✅ Configuration loaded successfully')"

clean:  ## Clean generated files (logs, plots, results, models)
	rm -rf src/results/logs/*.log
	rm -rf src/results/plots/*.png
	rm -rf src/results/analysis/*.csv
	rm -rf src/results/trained_models/*.pkl
	@echo "✅ Cleaned generated files from src/results/"

run-all:  ## Run complete pipeline (EDA + all models)
	python main.py

predict:  ## Make predictions with trained models (requires model name)
	@echo "Usage: make predict MODEL=decision_tree"
	@echo "Available models: decision_tree, logistic_regression, random_forest, svm"
	@if [ -n "$(MODEL)" ]; then \
		python src/scripts/predict_diabetes.py --model $(MODEL); \
	fi

show-results:  ## Show generated results directory structure
	@echo "📊 Results Directory Structure:"
	@if [ -d "src/results" ]; then \
		find src/results -type f -name "*.log" -o -name "*.png" -o -name "*.csv" -o -name "*.pkl" | head -20; \
	else \
		echo "No results found. Run 'make run-all' first."; \
	fi

# Quick setup for new environment
quick-setup: setup install check-config  ## Complete setup: directories + install + config check
	@echo "✅ Quick setup completed!"

# Development workflow
dev: clean check-config run-all  ## Development workflow: clean + check + run pipeline
	@echo "✅ Development pipeline completed!"

# Build and run (for build scripts)
build: setup install  ## Build: setup directories and install dependencies
	@echo "✅ Build completed!"

run: run-all  ## Alias for run-all
	@echo "✅ Pipeline execution completed!"