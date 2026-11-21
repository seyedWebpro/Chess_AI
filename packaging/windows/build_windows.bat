@echo off
REM build_windows.bat: Builds standalone Windows executable using PyInstaller

echo ========================================================
echo  Packaging Chess AI for Windows (x64)
echo ========================================================

cd /d "%~dp0\..\.."

REM Check PyInstaller
python -m pip install --upgrade pyinstaller

REM Build distribution with spec
python -m PyInstaller --clean --noconfirm packaging\chess_ai.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Standalone Windows executable built in dist\ChessAI\ChessAI.exe
    echo To build installer wizard, open packaging\windows\installer_windows.iss with Inno Setup.
) else (
    echo.
    echo [ERROR] PyInstaller build failed with code %ERRORLEVEL%
)

pause
