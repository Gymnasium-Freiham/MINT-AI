@echo off
REM Registry-Wert auslesen und in Variable speichern
FOR /F "tokens=3*" %%A IN ('reg query "HKCU\Software\LATIN AI" /v InstallDir') DO SET InstallPath=%%A

REM ZIP-Datei herunterladen, falls nicht vorhanden
IF NOT EXIST "%InstallPath%\executors\python.zip" (
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.14.0/python-3.14.0-embed-amd64.zip' -OutFile '%InstallPath%\\executors\\python.zip'"
)

REM ZIP-Datei entpacken, falls Zielordner nicht existiert
IF NOT EXIST "%InstallPath%\executors\python" (
    powershell -Command "$zip='%InstallPath%\\executors\\python.zip'; $dest='%InstallPath%\\executors\\python'; Expand-Archive -Path $zip -DestinationPath $dest -Force"
)

REM get-pip.py herunterladen
IF NOT EXIST "%InstallPath%\executors\get-pip.py" (
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%InstallPath%\\executors\\get-pip.py'"
)

REM python314._pth patchen: #import site entfernen
powershell -Command "(Get-Content '%InstallPath%\\executors\\python\\python314._pth') -replace '#import site','import site' | Set-Content '%InstallPath%\\executors\\python\\python314._pth'"

REM Installer .exe herunterladen
IF NOT EXIST "%InstallPath%\executors\python_installer.exe" (
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe' -OutFile '%InstallPath%\\executors\\python_installer.exe'"
)

REM Temporären Ordner für Installer-Extraktion erstellen
SET TempExtract=%InstallPath%\executors\python_installer_temp
IF NOT EXIST "%TempExtract%" mkdir "%TempExtract%"

SET Temp7z=%InstallPath%\executors\7z

REM Installer mit 7-Zip extrahieren
REM "%Temp7z%\7za.exe" x "%InstallPath%\executors\python_installer.exe" -o"%TempExtract%" -y

REM Prüfen, ob Include und libs existieren
REM IF EXIST "%TempExtract%\Include" (
    xcopy /E /Y "%TempExtract%\Include" "%InstallPath%\executors\python\Include"
REM ) ELSE (
REM     echo [FEHLER] Include-Ordner nicht gefunden. Installation abgebrochen.
REM     goto :eof
REM )

REM IF EXIST "%TempExtract%\libs" (
REM     xcopy /E /Y "%TempExtract%\libs" "%InstallPath%\executors\python\libs"
REM ) ELSE (
REM     echo [FEHLER] libs-Ordner nicht gefunden. Installation abgebrochen.
REM     goto :eof
REM )

REM pip installieren mit portabler Python
IF NOT EXIST "%InstallPath%\executors\python\Scripts\pip.exe" (
    "%InstallPath%\executors\python\python.exe" "%InstallPath%\executors\get-pip.py"
)

REM Setuptools und wheel installieren
"%InstallPath%\executors\python\python.exe" -m pip install setuptools wheel

REM simpleaudio installieren
REM "%InstallPath%\executors\python\python.exe" -m pip install simpleaudio
"%InstallPath%\executors\python\python.exe" -m pip install Cython==0.29.31
"%InstallPath%\executors\python\python.exe" -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
"%InstallPath%\executors\python\python.exe" -m pip install numpy
"%InstallPath%\executors\python\python.exe" -m pip install PyQt5
"%InstallPath%\executors\python\python.exe" -m pip install requests
"%InstallPath%\executors\python\python.exe" -m pip install matplotlib
"%InstallPath%\executors\python\python.exe" -m pip install groq

REM Temporäre Ordner löschen (optional)
rd /s /q "%TempExtract%"
rd /s /q "%Temp7z%"