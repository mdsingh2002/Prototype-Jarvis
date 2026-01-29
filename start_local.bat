@echo off
echo ==========================================
echo Hand Tracking Mouse Control - Local Mode
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

REM Install requirements if not already installed
echo Installing requirements...
pip install -r requirements_local.txt

echo.
echo ==========================================
echo Starting Hand Tracker...
echo ==========================================
echo.
echo Move mouse to corner of screen to emergency stop.
echo Press 'q' in the camera window to quit.
echo.

python src/hand_tracker_local.py
pause
