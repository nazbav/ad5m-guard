@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Сначала запустите install.bat
  pause
  exit /b
)
start "" ".venv\Scripts\pythonw.exe" launcher.pyw
