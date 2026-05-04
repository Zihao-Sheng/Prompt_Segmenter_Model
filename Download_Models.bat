@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

echo ============================================================
echo  Prompt Video Segmenter - Model Downloader
echo ============================================================
echo.
echo Select which pipeline you want to use:
echo.
echo   1. yolo11    - YOLO11-seg only          (~26 MB)   [recommended]
echo   2. gdino     - GroundingDINO + SAM2     (~820 MB)
echo   3. yolo_world- YOLO-World + SAM2        (~255 MB)
echo   4. full      - YOLO-World + SegFormer + SAM2 + MediaPipe (~285 MB)
echo   5. all       - Everything               (~1.1 GB)
echo.
set /p CHOICE="Enter pipeline name (default: yolo11): "
if "%CHOICE%"=="" set CHOICE=yolo11

set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Please run Install_Dependencies.bat first.
    pause
    exit /b 1
)

"%VENV_PYTHON%" "%PROJECT_ROOT%scripts\download_models.py" %CHOICE%
if errorlevel 1 (
    echo.
    echo Download failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Done. You can now run Launch_GUI.bat
pause
