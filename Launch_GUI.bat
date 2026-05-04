@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "PYTHON_EXE="
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE for %%I in (python.exe) do set "PYTHON_EXE=%%~$PATH:I"
if not defined PYTHON_EXE for %%I in (py.exe) do set "PYTHON_LAUNCHER=%%~$PATH:I"

if defined PYTHON_EXE (
    "%PYTHON_EXE%" -m src.gui_app
) else if defined PYTHON_LAUNCHER (
    "%PYTHON_LAUNCHER%" -3 -m src.gui_app
) else (
    echo Python 3.12+ was not found.
    echo Run Install_Dependencies.bat first, or install Python and try again.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo GUI exited with an error.
    pause
)
