#!/bin/bash
# Vytvoří virtuální prostředí a nainstaluje závislosti

echo "Vytváření virtuálního prostředí..."
python -m venv .venv

echo "Aktivace..."
source .venv/bin/activate 2>/dev/null || .venv\Scripts\activate

echo "Instalace závislostí..."
pip install --upgrade pip
pip install opencv-contrib-python pillow filterpy cffi numpy

echo "Hotovo! Aktivuj prostředí příkazem:"
echo "  source .venv/bin/activate  (Linux/Mac)"
echo "  .venv\Scripts\activate     (Windows)"