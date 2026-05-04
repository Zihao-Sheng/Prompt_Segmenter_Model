@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ============================================================
echo  Downloading model weights...
echo ============================================================
echo.

set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Please run Install_Dependencies.bat first.
    pause
    exit /b 1
)

"%VENV_PYTHON%" "%PROJECT_ROOT%scripts\download_models.py"
if errorlevel 1 (
    echo Download failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Models downloaded successfully.
pause
