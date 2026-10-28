@echo off
cd /d "%~dp0"
set "MLCLOUD_PYTHON=C:\Users\domin\.conda\envs\py310\python.exe"
if not exist "%MLCLOUD_PYTHON%" set "MLCLOUD_PYTHON=python"
"%MLCLOUD_PYTHON%" server.py
if errorlevel 1 pause
