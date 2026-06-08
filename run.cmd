@echo off
REM ===================================================================
REM  run.cmd — spustí video_anonymizer.py (single-file CLI + wizard).
REM
REM  Bez argumentů → interaktivní průvodce.
REM  S argumenty  → přímé spuštění (předá vše do video_anonymizer.py).
REM
REM  Příklady:
REM     run                                         (wizard)
REM     run video.mp4                               (defaulty)
REM     run video.mp4 --tracker kcf --blur-method gaussian
REM     run --wizard                               (vynutí wizard)
REM ===================================================================

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run.cmd] CHYBA: .venv neexistuje.
    exit /b 1
)

call .venv\Scripts\activate.bat >nul
if errorlevel 1 (
    echo [run.cmd] CHYBA: nelze aktivovat .venv
    exit /b 1
)

python video_anonymizer.py %*
endlocal
