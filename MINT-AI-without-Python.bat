@echo off

REM Registry-Schlüssel auslesen, um InstallDir zu ermitteln
set "InstallDir="
for /f "usebackq tokens=2*" %%A in (`reg query "HKCU\Software\MINT-AI" /v InstallDir 2^>nul`) do set "InstallDir=%%B"

REM Überprüfen, ob InstallDir gesetzt wurde
if "%InstallDir%"=="" (
    echo Installationsverzeichnis nicht gefunden.
    pause
    exit /b
)

REM Virtuelle Umgebung erstellen, falls nicht vorhanden
if not exist "%InstallDir%\venv\Scripts\python.exe" (
    echo Virtuelle Umgebung wird erstellt...
    "%InstallDir%\python\python.exe" -m venv "%InstallDir%\venv"
)

REM Virtuelle Umgebung aktivieren und launcher.py ausführen
"%InstallDir%\venv\Scripts\python.exe" "%InstallDir%\launcher.py"