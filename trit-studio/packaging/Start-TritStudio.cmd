@echo off
cd /d "%~dp0"
if not exist "TritStudio.exe" (
  echo TritStudio.exe is missing. Merge update CONTENTS into the existing compatible installation. An update is not standalone.
  pause
  exit /b 1
)
start "" "%~dp0TritStudio.exe"
