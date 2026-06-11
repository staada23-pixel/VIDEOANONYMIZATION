@echo off
setlocal

set "PR=%~dp0"
if "%PR:~-1%"=="\" set "PR=%PR:~0,-1%"

set "LPM=C:\Users\face\Desktop\Praxe 2026\EYEDEA PROJECT\LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2\wrappers\python"
if not exist "%LPM%\lpm.py" set "LPM="

if defined LPM (
    set "PP=%PR%\src;%LPM%"
) else (
    set "PP=%PR%\src"
)

set "PYTHONPATH=%PP%"
set "PYTHONIOENCODING=utf-8"

REM Zustan v PR (kvuli relativnim cestam k souborum), ale pridej src do PYTHONPATH
cd /d "%PR%"

python -m video_anonymizer %*
