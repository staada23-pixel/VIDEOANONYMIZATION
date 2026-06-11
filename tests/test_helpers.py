"""Shared helpers for tests and local smoke runs."""
from __future__ import annotations

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, ".."))

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from video_anonymizer.video_catalog import discover_videos, primary_test_video


def resolve_test_video(prefer_bodycam: bool = False) -> str:
    """
    Vrati realne testovaci video z tests/ (ne synteticky demo.mp4).
    prefer_bodycam=True zkusi bodycam z LPM SDK misto tests/.
    """
    videos = discover_videos(_PROJECT_ROOT)

    if prefer_bodycam:
        for _label, path in videos:
            if "bodycam" in path.lower() or "capture of publicly" in path.lower():
                return path

    for _label, path in videos:
        if path.startswith(os.path.join(_PROJECT_ROOT, "tests")):
            return path

    video = primary_test_video(_PROJECT_ROOT)
    if video:
        return video

    raise FileNotFoundError(
        "Zadne testovaci video v tests/. "
        "Vloz .mp4 do tests/ (napr. WhatsApp video)."
    )
