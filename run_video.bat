@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo Перетащите на этот файл видео печати, папку или zip с кадрами.
  pause
  exit /b
)
.venv\Scripts\python run_video.py %*
pause
