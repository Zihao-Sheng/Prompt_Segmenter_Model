@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ============================================================
echo  Prompt Video Segmenter - Dependency Installer
echo ============================================================
echo.
echo Prerequisites:
echo   1. Python 3.12+  ^(https://www.python.org/downloads/^)
echo      - During install, check "Add python.exe to PATH"
echo   2. NVIDIA GPU driver supporting CUDA 12.8+
echo      ^(driver version 525+ recommended^)
echo   3. ~6 GB free disk space
echo.
echo If you do not have an NVIDIA GPU, the app will run on CPU
echo ^(much slower, but still functional^).
echo.
pause

set "PYTHON_EXE="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE for %%I in (python.exe) do set "PYTHON_EXE=%%~$PATH:I"
if not defined PYTHON_EXE for %%I in (py.exe) do set "PYTHON_LAUNCHER=%%~$PATH:I"

if not exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    if defined PYTHON_EXE (
        "%PYTHON_EXE%" -m venv "%PROJECT_ROOT%.venv"
    ) else if defined PYTHON_LAUNCHER (
        "%PYTHON_LAUNCHER%" -3 -m venv "%PROJECT_ROOT%.venv"
    ) else (
        echo Python 3.12+ was not found.
        echo Please install Python first: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo Installing PyTorch with CUDA 12.8 support...
"%VENV_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
if errorlevel 1 goto :fail

echo.
echo Dependencies installed successfully.
echo You can now run Launch_GUI.bat
pause
exit /b 0

:fail
echo.
echo Dependency installation failed.
echo If torch GPU wheels are needed on another machine, install torch separately first and rerun this script.
pause
exit /b 1
