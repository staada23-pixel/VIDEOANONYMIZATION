#!/usr/bin/env python3
"""
Spusteni z korenove slozky projektu:

    python cli.py
    python cli.py --input tests\\video.mp4 --detector face
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from video_anonymizer.bootstrap import setup_environment
from video_anonymizer.cli import main

setup_environment(ROOT)

if __name__ == "__main__":
    raise SystemExit(main())

