@echo off
REM Setup script for AI Interview Intelligence System (Windows)

echo ===============================================================
echo    AI Interview Intelligence System - Setup Script (Windows)
echo ===============================================================
echo.

REM Check Python version
echo 1. Checking Python version...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo    X Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo    + Python found
echo.

REM Check FFmpeg
echo 2. Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo    ! FFmpeg not found. Please install from:
    echo      https://ffmpeg.org/download.html
    pause
)
echo.

REM Create virtual environment
echo 3. Creating virtual environment...
if exist venv (
    echo    ! Virtual environment already exists
    choice /M "Recreate virtual environment"
    if errorlevel 2 goto skip_venv
    rmdir /s /q venv
)
python -m venv venv
echo    + Virtual environment created
:skip_venv
echo.

REM Activate virtual environment
echo 4. Activating virtual environment...
call venv\Scripts\activate.bat
echo    + Virtual environment activated
echo.

REM Upgrade pip
echo 5. Upgrading pip...
python -m pip install --upgrade pip --quiet
echo    + pip upgraded
echo.

REM Install dependencies
echo 6. Installing dependencies (this may take several minutes)...
pip install -r requirements.txt --quiet
if %ERRORLEVEL% EQU 0 (
    echo    + All dependencies installed
) else (
    echo    X Some dependencies failed
    echo    Try running: pip install -r requirements.txt
)
echo.

REM Create directories
echo 7. Creating directories...
mkdir data\sample_videos 2>nul
mkdir data\models_cache 2>nul
mkdir outputs 2>nul
mkdir models 2>nul
echo    + Directories created
echo.

REM Run system check
echo 8. Running system check...
python demo.py
echo.

echo ===============================================================
echo    + Setup Complete!
echo ===============================================================
echo.
echo Next steps:
echo   1. Activate: venv\Scripts\activate.bat
echo   2. Launch: streamlit run app.py
echo   3. Or run: python demo.py
echo.
echo For help, see README.md and docs\
echo.
pause
