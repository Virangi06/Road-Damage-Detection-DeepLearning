@echo off
echo ============================================================
echo   Road Damage AI -- React Frontend
echo   http://localhost:5173
echo   (Requires API to be running on port 5000)
echo ============================================================
cd /d %~dp0\frontend
npm run dev
pause
