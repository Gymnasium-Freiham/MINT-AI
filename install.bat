@echo off
setlocal

:: Verzeichnis, in dem die Batch-Datei liegt
set "SCRIPT_DIR=%~dp0"

:: Zielverzeichnis
set "TARGET=%USERPROFILE%\MINT-AI"

:: Zielverzeichnis erstellen
mkdir "%TARGET%"

:: ZIP entpacken aus dem Verzeichnis, wo die Batch-Datei liegt
powershell -Command "Expand-Archive -Path '%SCRIPT_DIR%mintai.zip' -DestinationPath '%TARGET%' -Force"

:: Registry-Eintrag setzen
reg add "HKCU\Software\MINT-AI" /v InstallDir /t REG_SZ /d "%TARGET%" /f

echo Installation abgeschlossen.
endlocal
