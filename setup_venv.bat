@echo off
echo Vytvarime virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
echo Instalujeme zavislosti...
pip install --upgrade pip
pip install opencv-contrib-python numpy Pillow cffi PyYAML
echo.
echo Hotovo! Aktivuj environment:
echo   .venv\Scripts\activate.bat
echo Pak spust:
echo   python -m video_anonymizer --input video.mp4
pause
