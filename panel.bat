@echo off
rem Launch the HC control panel from anywhere (double-click or run by full path).
rem %~dp0 = this file's folder (project root), so cwd never matters.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m dh.webapp.server
pause
