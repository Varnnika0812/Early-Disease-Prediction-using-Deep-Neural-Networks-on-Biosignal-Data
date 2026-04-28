@echo off
setlocal

cd /d "%~dp0"
set "PROJECT_DRIVE=Z:"

if not exist "%PROJECT_DRIVE%\" (
    subst %PROJECT_DRIVE% "%CD%" >nul 2>nul
    if errorlevel 1 (
        echo Could not map %CD% to %PROJECT_DRIVE%. Edit setup_environment.bat and choose a free drive letter.
        pause
        exit /b 1
    )
)

cd /d %PROJECT_DRIVE%\

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher was not found. Install Python 3.11, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv311\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv311
    if errorlevel 1 (
        echo Could not create the virtual environment with Python 3.11.
        pause
        exit /b 1
    )
)

echo Installing required packages into .venv311...
".venv311\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    pause
    exit /b 1
)

".venv311\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo Setup complete. Use run_project.bat to start the app.
pause
