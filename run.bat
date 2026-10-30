@echo off
rem Hotbench on http://localhost:8300 (settings in .env, see .env.example)
cd /d "%~dp0"
.venv\Scripts\python app.py
