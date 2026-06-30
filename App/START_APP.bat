@echo off
title AOI Studio

set "APP_ROOT=%~dp0"
set "PROJECT_ROOT=%APP_ROOT%\.."
set "PY=%PROJECT_ROOT%\venv\Scripts\python.exe"
set "SRC_DIR=%APP_ROOT%\src"

cd /d "%APP_ROOT%"

if not exist "%PY%" (
    echo ERROR: venv not found. Expected: %PROJECT_ROOT%\venv\
    echo Install Python 3.11 and recreate the venv before running the app.
    pause
    exit /b 1
)

"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo ERROR: The existing venv Python cannot start.
    echo Recreate the venv with Python 3.11, then install App\requirements-qt.txt.
    pause
    exit /b 1
)

"%PY%" -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo ERROR: PySide6 is not installed in the venv.
    echo Run: "%PY%" -m pip install -r "%APP_ROOT%\requirements-qt.txt"
    pause
    exit /b 1
)

"%PY%" "%SRC_DIR%\app.py"

echo.
echo App closed.
pause
