@echo off
title Road Damage AI -- Flask API (port 5000)
echo ============================================================
echo   Road Damage AI -- Flask API Backend
echo   URL: http://localhost:5000
echo.
echo   IMPORTANT: Keep this window open while using the app.
echo   Models will load for ~30-60 seconds before the first
echo   request can be handled.
echo ============================================================
echo.
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Verify Python is available
python --version

echo.
echo Starting API server...
echo.
python api.py

echo.
echo API server stopped. Press any key to exit.
pause >nul
