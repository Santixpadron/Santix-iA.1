@echo off
cd /d "%~dp0"
echo Iniciando detector de residuos web...
set PYTHONIOENCODING=utf-8
python web_app.py
pause
