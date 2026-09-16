@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Check-Update.ps1" -Root "%~dp0."
set "code=%errorlevel%"
pause
exit /b %code%
