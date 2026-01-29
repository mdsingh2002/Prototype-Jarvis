@echo off
echo ==========================================
echo Hand Tracking Mouse Control - Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH!
    echo Please install Python 3.x and try again.
    pause
    exit /b 1
)

REM Install host requirements if not already installed
echo Installing host requirements...
pip install -r host_requirements.txt

echo.
echo ==========================================
echo Starting Mouse Server...
echo ==========================================
echo.
echo After the server starts, run 'docker-compose up' in another terminal.
echo.

python src/mouse_server.py
