@echo off
setlocal

cd /d "%~dp0"

echo Starting Memory Auto-Label GUI...
python run_memory_autolabel_gui.py

if errorlevel 1 (
    echo.
    echo GUI exited with an error.
    echo If Python or PySide6 is missing, run:
    echo   python -m pip install PySide6 opencv-python numpy pillow
    echo.
    pause
)

endlocal
