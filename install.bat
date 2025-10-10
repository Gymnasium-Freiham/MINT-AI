@echo off
echo Starte LATIN-AI Installation...

REM Prüfen ob git verfügbar ist
where git >nul 2>nul
IF ERRORLEVEL 1 (
    echo Fehler: Git ist nicht installiert oder nicht im PATH.
    exit /b 1
)

REM In das aktuelle Verzeichnis wechseln
cd /d "%~dp0"

REM Temporäres Verzeichnis zum Klonen
set TMPCLONE=%TEMP%\mintai_clone_tmp

REM Vorheriges temporäres Verzeichnis löschen, falls vorhanden
IF EXIST "%TMPCLONE%" (
    rmdir /s /q "%TMPCLONE%"
)

REM Repository klonen
echo Klone LATIN-AI nach temporärem Verzeichnis...
git clone --depth 1 --single-branch --branch main https://github.com/Gymnasium-Freiham/LATIN-AI.git "%TMPCLONE%"
IF ERRORLEVEL 1 (
    echo Fehler beim Klonen.
    exit /b 1
)

REM Inhalte ins aktuelle Verzeichnis kopieren
xcopy "%TMPCLONE%\*" "%CD%\" /E /H /Y >nul

REM Temporäres Verzeichnis löschen
rmdir /s /q "%TMPCLONE%"

echo Installation abgeschlossen
exit /b 0
