@echo off
title NASEA Installer
color 0A

echo ============================================
echo           NASEA Auto-Installer
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Opening Microsoft Store to install Python...
    echo.
    echo After installing Python, run this script again.
    echo.
    start ms-windows-store://pdp/?productid=9NRWMJP3717K
    pause
    exit
)

echo [OK] Python found!
echo.

:: Create virtual environment
echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit
)
echo [OK] Virtual environment created!
echo.

:: Activate and install
echo Installing dependencies (this may take a minute)...
call .venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit
)
echo [OK] Dependencies installed!
echo.

:: Install nasea package
pip install -e . >nul 2>&1
echo [OK] NASEA installed!
echo.

:: Create launcher
echo Creating launcher...
(
echo @echo off
echo cd /d "%%~dp0"
echo call .venv\Scripts\activate.bat
echo nasea
) > RUN_NASEA.bat
echo [OK] Launcher created!

:: Create .env if not exists
if not exist .env (
    echo Creating .env file...
    echo # NASEA Configuration - Add your API key below > .env
    echo # You only need ONE of these keys >> .env
    echo. >> .env
    echo VENICE_API_KEY=your_key_here >> .env
    echo # OPENAI_API_KEY=your_key_here >> .env
    echo # KIMI_API_KEY=your_key_here >> .env
    echo.
    echo [OK] .env file created!
    echo.
    echo Opening .env in Notepad - ADD YOUR API KEY and save!
    start notepad .env
)
echo.

echo ============================================
echo       Installation Complete!
echo ============================================
echo.
echo IMPORTANT: Edit .env file and add your API key!
echo.
echo Usage:
echo   nasea generate "create a snake game"
echo   nasea generate "build a todo app"
echo   nasea --help
echo.
echo To run: double-click RUN_NASEA.bat
echo.
pause
