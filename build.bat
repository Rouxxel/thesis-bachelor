@echo off
REM Diabetes Prediction ML Pipeline - Build Script (Windows)
REM This script sets up the environment and runs the ML pipeline

setlocal enabledelayedexpansion

goto :parse_args

REM Function to print status messages
:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM Main build function
:main
echo 🧠 Diabetes Prediction ML Pipeline - Build Script
echo ==================================================

REM Check Python installation
call :print_status "Checking Python installation..."

REM Try python command first, then py
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
    ) else (
        call :print_error "Python is not installed or not in PATH"
        echo Please install Python 3.7+ from https://python.org
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
call :print_success "Found: !PYTHON_VERSION!"

REM Check Python version (require 3.7+)
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Python 3.7 or higher is required"
    pause
    exit /b 1
)

REM Setup directories
call :print_status "Setting up project directories..."
%PYTHON_CMD% setup.py
if %errorlevel% neq 0 (
    call :print_error "Failed to setup directories"
    pause
    exit /b 1
)

REM Install dependencies
call :print_status "Installing Python dependencies..."
if exist requirements.txt (
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %errorlevel% equ 0 (
        call :print_success "Dependencies installed successfully"
    ) else (
        call :print_error "Failed to install dependencies"
        pause
        exit /b 1
    )
) else (
    call :print_warning "requirements.txt not found, skipping dependency installation"
)

REM Check configuration
call :print_status "Checking configuration..."
%PYTHON_CMD% -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.') / 'src')); from src.core.config_loader import config_loader; print('Configuration loaded successfully')"
if %errorlevel% neq 0 (
    call :print_error "Configuration check failed"
    pause
    exit /b 1
)

REM Check dataset
call :print_status "Checking dataset availability..."
if exist "dataset\diabetes_binary_health_indicators_BRFSS2015.csv" (
    call :print_success "Dataset found"
) else (
    call :print_warning "Dataset not found at dataset\diabetes_binary_health_indicators_BRFSS2015.csv"
    call :print_warning "Please ensure the dataset is in the correct location"
)

call :print_success "Build completed successfully!"
echo.
echo 📋 Next steps:
echo   1. Run the complete pipeline: build.bat run
echo   2. Or run manually: python main.py
echo   3. Make predictions: python src\scripts\predict_diabetes.py --model decision_tree
echo   4. Check results in: src\results\
goto :eof

REM Run pipeline function
:run_pipeline
call :print_status "Running ML pipeline..."
%PYTHON_CMD% main.py
if %errorlevel% equ 0 (
    call :print_success "Pipeline execution completed!"
    echo.
    call :print_status "Results available in:"
    echo   📊 Logs: src\results\logs\
    echo   📈 Plots: src\results\plots\
    echo   🤖 Models: src\results\trained_models\
    echo   📋 Analysis: src\results\analysis\
) else (
    call :print_error "Pipeline execution failed"
    echo Check the logs in src\results\logs\ for details
    pause
    exit /b 1
)
goto :eof

REM Clean function
:clean
call :print_status "Cleaning generated files..."
if exist "src\results\logs\*.log" del /q "src\results\logs\*.log" >nul 2>&1
if exist "src\results\plots\*.png" del /q "src\results\plots\*.png" >nul 2>&1
if exist "src\results\analysis\*.csv" del /q "src\results\analysis\*.csv" >nul 2>&1
if exist "src\results\trained_models\*.pkl" del /q "src\results\trained_models\*.pkl" >nul 2>&1
call :print_success "Cleaned generated files"
goto :eof

REM Help function
:show_help
echo 🧠 Diabetes Prediction ML Pipeline - Build Script
echo ==================================================
echo.
echo Usage: build.bat [command]
echo.
echo Commands:
echo   (no args)  Setup environment and install dependencies
echo   run        Setup + run the complete ML pipeline
echo   clean      Clean all generated files
echo   help       Show this help message
echo.
echo Examples:
echo   build.bat          # Setup only
echo   build.bat run      # Setup and run pipeline
echo   build.bat clean    # Clean generated files
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" (
    call :main
) else if /i "%~1"=="run" (
    call :main
    if %errorlevel% equ 0 call :run_pipeline
) else if /i "%~1"=="clean" (
    call :clean
) else if /i "%~1"=="help" (
    call :show_help
) else if "%~1"=="-h" (
    call :show_help
) else if "%~1"=="--help" (
    call :show_help
) else (
    call :print_error "Unknown command: %~1"
    call :show_help
    pause
    exit /b 1
)

endlocal