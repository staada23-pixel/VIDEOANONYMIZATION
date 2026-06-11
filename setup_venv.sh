#!/bin/bash
set -e
echo "Vytváříme virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "Instalujeme závislosti..."
pip install --upgrade pip
pip install opencv-contrib-python numpy Pillow cffi PyYAML
echo ""
echo "Hotovo! Aktivuj environment:"
echo "  source .venv/bin/activate"
echo "Pak spusť:"
echo "  python -m video_anonymizer --input video.mp4"
