@echo off
echo ============================================================
echo   Road Damage AI -- Flask API Backend
echo   http://localhost:5000
echo ============================================================
cd /d %~dp0
call .venv\Scripts\activate.bat
python api.py
pause
