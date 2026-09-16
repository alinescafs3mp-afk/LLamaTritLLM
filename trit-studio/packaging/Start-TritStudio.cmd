@echo off
cd /d "%~dp0"
if not exist "TritStudio.exe" (
  echo TritStudio.exe is missing. Extract the ENTIRE portable ZIP first.
  pause
  exit /b 1
)
start "" "%~dp0TritStudio.exe"
