@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Check-laptop.ps1"
set "result=%ERRORLEVEL%"
echo.
echo Diagnostics exit code: %result%
pause
exit /b %result%
