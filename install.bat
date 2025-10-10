@echo off
echo Starte MINT-AI Installation...

REM Prüfen ob git verfügbar ist
where git >nul 2>nul
IF ERRORLEVEL 1 (
    echo Fehler: Git ist nicht installiert oder nicht im PATH.
    exit /b 1
)

REM In das aktuelle Verzeichnis wechseln
cd /d "%~dp0"

REM Repository direkt in dieses Verzeichnis klonen
git clone --depth 1 --single-branch --branch main https://github.com/Gymnasium-Freiham/MINT-AI.git . >nul 2>&1
IF ERRORLEVEL 1 (
    echo Fehler beim Klonen.
    exit /b 1
)

echo Installation abgeschlossen
exit /b 0
