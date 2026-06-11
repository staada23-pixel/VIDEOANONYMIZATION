"""Vyhledani vstupnich videi v projektu."""
from __future__ import annotations

import os

_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
_SKIP_BASENAMES = {
    "demo_anonymized.mp4",
}
_SKIP_PREFIXES = ("test_",)
_SKIP_CONTAINS = ("_anonymized",)


def _is_user_video(filename: str) -> bool:
    low = filename.lower()
    if low in {s.lower() for s in _SKIP_BASENAMES}:
        return False
    if any(low.startswith(p) for p in _SKIP_PREFIXES):
        return False
    if any(part in low for part in _SKIP_CONTAINS):
        return False
    return low.endswith(_VIDEO_EXTS)


def _bodycam_candidates(project_root: str) -> list[str]:
    return [
        os.path.join(
            project_root, "..",
            "LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2",
            "wrappers", "python",
            "capture of publicly available bodycam footage.mp4",
        ),
        os.path.join(
            r"C:\Users\face\Desktop\Praxe 2026\EYEDEA PROJECT",
            "LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2",
            "wrappers", "python",
            "capture of publicly available bodycam footage.mp4",
        ),
        os.path.join(
            project_root, "externals", "LPM", "wrappers", "python",
            "capture of publicly available bodycam footage.mp4",
        ),
    ]


def discover_videos(project_root: str) -> list[tuple[str, str]]:
    """
    Vrat [(label, abs_path), ...].
    Realna videa z tests/input/videos prvni (nej novejsi), pak bodycam z LPM.
    """
    found: list[tuple[str, str, float]] = []

    for sub in ("tests", "input", "videos"):
        folder = os.path.join(project_root, sub)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not _is_user_video(name):
                continue
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                found.append((name, os.path.abspath(path), os.path.getmtime(path)))

    found.sort(key=lambda x: x[2], reverse=True)

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, path, _ in found:
        if path not in seen:
            seen.add(path)
            results.append((label, path))

    for candidate in _bodycam_candidates(project_root):
        if os.path.isfile(candidate):
            path = os.path.abspath(candidate)
            if path not in seen:
                seen.add(path)
                results.append((os.path.basename(candidate), path))
            break

    return results


def primary_test_video(project_root: str) -> str | None:
    """Nejlepsi video pro testy — prvni z katalogu, ne synteticky demo."""
    videos = discover_videos(project_root)
    return videos[0][1] if videos else None
