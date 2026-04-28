@echo off
setlocal

cd /d "%~dp0"
set "PROJECT_DRIVE=Z:"

if not exist "%PROJECT_DRIVE%\" (
    subst %PROJECT_DRIVE% "%CD%" >nul 2>nul
    if errorlevel 1 (
        echo Could not map %CD% to %PROJECT_DRIVE%. Edit run_project.bat and choose a free drive letter.
        pause
        exit /b 1
    )
)

cd /d %PROJECT_DRIVE%\

if not exist ".venv311\Scripts\python.exe" (
    echo Project environment is not ready.
    echo Run setup_environment.bat once, then run this file again.
    pause
    exit /b 1
)

".venv311\Scripts\python.exe" -c "import tensorflow, numpy, pandas, matplotlib, scipy, sklearn" >nul 2>nul
if errorlevel 1 (
    echo Required packages are missing from .venv311.
    echo Run setup_environment.bat once, then run this file again.
    pause
    exit /b 1
)

".venv311\Scripts\python.exe" "mini project .py"
