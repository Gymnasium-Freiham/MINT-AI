@echo off
<<<<<<< HEAD
echo Starte LATIN-AI Installation...
=======
echo Starte MINT-AI Installation...
>>>>>>> 46ee4ab922b63c1a0a1eea79b1a004668e919ef2

REM Prüfen ob git verfügbar ist
where git >nul 2>nul
IF ERRORLEVEL 1 (
    echo Fehler: Git ist nicht installiert oder nicht im PATH.
    exit /b 1
)

REM In das aktuelle Verzeichnis wechseln
cd /d "%~dp0"

<<<<<<< HEAD
REM Temporäres Verzeichnis zum Klonen
set TMPCLONE=%TEMP%\mintai_clone_tmp

REM Vorheriges temporäres Verzeichnis löschen, falls vorhanden
IF EXIST "%TMPCLONE%" (
    rmdir /s /q "%TMPCLONE%"
)

REM Repository klonen
echo Klone LATIN-AI nach temporärem Verzeichnis...
git clone --depth 1 --single-branch --branch main https://github.com/Gymnasium-Freiham/MINT-AI.git "%TMPCLONE%"
=======
REM Repository direkt in dieses Verzeichnis klonen
git clone --depth 1 --single-branch --branch main https://github.com/Gymnasium-Freiham/MINT-AI.git . >nul 2>&1
>>>>>>> 46ee4ab922b63c1a0a1eea79b1a004668e919ef2
IF ERRORLEVEL 1 (
    echo Fehler beim Klonen.
    exit /b 1
)

<<<<<<< HEAD
REM Inhalte ins aktuelle Verzeichnis kopieren
xcopy "%TMPCLONE%\*" "%CD%\" /E /H /Y >nul

REM Temporäres Verzeichnis löschen
rmdir /s /q "%TMPCLONE%"

echo Installation abgeschlossen
exit /b 0
=======
echo Installation abgeschlossen
exit /b 0
>>>>>>> 46ee4ab922b63c1a0a1eea79b1a004668e919ef2
