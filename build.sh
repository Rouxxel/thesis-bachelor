#!/bin/bash

# Diabetes Prediction ML Pipeline - Build Script (Unix/Linux/macOS)
# This script sets up the environment and runs the ML pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main build function
main() {
    echo "🧠 Diabetes Prediction ML Pipeline - Build Script"
    echo "=================================================="
    
    # Check Python installation
    print_status "Checking Python installation..."
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_success "Found: $PYTHON_VERSION"
    
    # Check Python version (require 3.7+)
    PYTHON_VERSION_CHECK=$($PYTHON_CMD -c "import sys; print(sys.version_info >= (3, 7))")
    if [ "$PYTHON_VERSION_CHECK" != "True" ]; then
        print_error "Python 3.7 or higher is required"
        exit 1
    fi
    
    # Setup directories
    print_status "Setting up project directories..."
    $PYTHON_CMD setup.py
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_warning "requirements.txt not found, skipping dependency installation"
    fi
    
    # Check configuration
    print_status "Checking configuration..."
    $PYTHON_CMD -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))
from src.core.config_loader import config_loader
print('✅ Configuration loaded successfully')
"
    
    # Check dataset
    print_status "Checking dataset availability..."
    if [ -f "dataset/diabetes_binary_health_indicators_BRFSS2015.csv" ]; then
        print_success "Dataset found"
    else
        print_warning "Dataset not found at dataset/diabetes_binary_health_indicators_BRFSS2015.csv"
        print_warning "Please ensure the dataset is in the correct location"
    fi
    
    print_success "Build completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "  1. Run the complete pipeline: ./build.sh run"
    echo "  2. Or run manually: python main.py"
    echo "  3. Make predictions: python src/scripts/predict_diabetes.py --model decision_tree"
    echo "  4. Check results in: src/results/"
}

# Run pipeline function
run_pipeline() {
    print_status "Running ML pipeline..."
    $PYTHON_CMD main.py
    print_success "Pipeline execution completed!"
    
    print_status "Results available in:"
    echo "  📊 Logs: src/results/logs/"
    echo "  📈 Plots: src/results/plots/"
    echo "  🤖 Models: src/results/trained_models/"
    echo "  📋 Analysis: src/results/analysis/"
}

# Clean function
clean() {
    print_status "Cleaning generated files..."
    rm -rf src/results/logs/*.log 2>/dev/null || true
    rm -rf src/results/plots/*.png 2>/dev/null || true
    rm -rf src/results/analysis/*.csv 2>/dev/null || true
    rm -rf src/results/trained_models/*.pkl 2>/dev/null || true
    print_success "Cleaned generated files"
}

# Help function
show_help() {
    echo "🧠 Diabetes Prediction ML Pipeline - Build Script"
    echo "=================================================="
    echo ""
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Commands:"
    echo "  (no args)  Setup environment and install dependencies"
    echo "  run        Setup + run the complete ML pipeline"
    echo "  clean      Clean all generated files"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./build.sh          # Setup only"
    echo "  ./build.sh run      # Setup and run pipeline"
    echo "  ./build.sh clean    # Clean generated files"
}

# Parse command line arguments
case "${1:-}" in
    "run")
        main
        run_pipeline
        ;;
    "clean")
        clean
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac