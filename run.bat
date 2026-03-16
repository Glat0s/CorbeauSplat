@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"
echo Working directory: %CD%

:: ============================================================
:: Phase 1: Update Check
:: ============================================================
if exist ".git" (
    echo --- Phase 1: Checking for updates ---
    git fetch >nul 2>&1
    for /f %%i in ('git rev-list --count HEAD..@{u} 2^>nul') do set BEHIND=%%i
    if defined BEHIND (
        if !BEHIND! GTR 0 (
            echo ^>^>^> A new version is available (!BEHIND! commits behind^).
            set /p REPLY=">>> Update now? (y/n): "
            if /i "!REPLY!"=="y" (
                echo Updating...
                git pull
                echo Update complete.
            ) else (
                echo Update skipped.
            )
        ) else (
            echo Software is up to date.
        )
    )
) else (
    echo --- Phase 1: Skipping update check (not a git repository) ---
)

:: ============================================================
:: Phase 2: Python environment
:: ============================================================
echo --- Phase 2: Environment configuration ---
set VENV_DIR=.venv
set PYTHON_CMD=%VENV_DIR%\Scripts\python.exe

if not exist "%PYTHON_CMD%" (
    echo Creating virtual environment...
    if exist "%VENV_DIR%" (
        echo Venv corrupted. Rebuilding...
        rmdir /s /q "%VENV_DIR%"
    )

    :: Try python3 then python
    where python3 >nul 2>&1 && (
        set SELECTED_PY=python3
        goto :venv_create
    )
    where python >nul 2>&1 && (
        set SELECTED_PY=python
        goto :venv_create
    )
    echo ERROR: Python not found. Please install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1

    :venv_create
    echo Using: !SELECTED_PY!
    !SELECTED_PY! -m venv %VENV_DIR%
    echo Virtual environment created.
)

echo Using environment: %PYTHON_CMD%
for /f "tokens=*" %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do echo %%v

:: Integrity check
"%PYTHON_CMD%" -c "import json, os, sys" >nul 2>&1
if errorlevel 1 (
    echo FAILURE: Python environment unstable. Forcing rebuild...
    rmdir /s /q "%VENV_DIR%"
    call "%~f0" %*
    exit /b
)
echo Python environment verified.

:: ============================================================
:: Phase 3: Dependency sync
:: ============================================================
echo --- Phase 3: Synchronizing dependencies ---
"%PYTHON_CMD%" -m pip install --upgrade pip --quiet

if exist "requirements.lock" (
    set DEP_FILE=requirements.lock
    echo Found lockfile: %DEP_FILE%
) else (
    set DEP_FILE=requirements.txt
    echo Found dependency list: %DEP_FILE%
)

echo Verifying installed packages...
"%PYTHON_CMD%" -m pip install -r %DEP_FILE% --quiet
if errorlevel 1 (
    echo Silent installation failed. Retrying with output...
    "%PYTHON_CMD%" -m pip install -r %DEP_FILE%
)
echo Dependencies synchronized.

:: PyQt6 check
"%PYTHON_CMD%" -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Installing PyQt6...
    "%PYTHON_CMD%" -m pip install PyQt6
)

:: send2trash check
"%PYTHON_CMD%" -c "import send2trash" >nul 2>&1
if errorlevel 1 (
    echo Installing send2trash...
    "%PYTHON_CMD%" -m pip install send2trash
)

:: opencv check
"%PYTHON_CMD%" -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing opencv-python...
    "%PYTHON_CMD%" -m pip install opencv-python
)

:: ============================================================
:: Phase 4: Engine verification
:: ============================================================
echo --- Phase 4: Verifying engines and external binaries ---
"%PYTHON_CMD%" -m app.scripts.setup_dependencies --startup
echo System check complete.

:: Architecture info
echo Architecture: Windows (CUDA/RTX optimizations active if NVIDIA GPU present)
nvidia-smi >nul 2>&1 && echo NVIDIA GPU detected - CUDA acceleration available.

:: ============================================================
:: Phase 5: Launch
:: ============================================================
echo --- Phase 5: Launching CorbeauSplat ---
echo ------------------------------------------------
"%PYTHON_CMD%" main.py %*

endlocal
