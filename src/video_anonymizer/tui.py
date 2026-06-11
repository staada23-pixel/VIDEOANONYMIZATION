"""Interaktivni TUI menu — vyber cislem, zadne rucni cesty."""
from __future__ import annotations

import os
import re
import sys

from .video_catalog import discover_videos

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

DETECTORS = [
    ("auto",  "Auto (LPM pokud HASP, jinak face DNN)"),
    ("lpm",   "LPM (Eyedea — vyzaduje HASP dongle)"),
    ("face",  "Face DNN YuNet (obliceje, doporuceno)"),
    ("hog",   "HOG (cele postavy)"),
    ("haar",  "Haar Cascade (obliceje, fallback)"),
]

TRACKERS = [
    ("KCF",   "KCF Gaussian kernel (template-matching fallback)"),
    ("CSRT",  "CSRT (OpenCV) — pomalejsi, presnejsi"),
    ("MIL",   "MIL (OpenCV) — rychlejsi, mene presny"),
    ("VIT",   "ViTTrack (Transformer) — SOTA, nejpřesnější"),
]

ANON_METHODS = [
    ("none",   "Bez anonymizace (jen boxy)"),
    ("mosaic", "Mosaic / pixelize (doporuceno)"),
    ("blur",   "Gaussovo rozmazani"),
    ("black",  "Cerny obdelnik"),
    ("solid",  "Jednobarevny obdelnik"),
]


def _print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def _quit(raw: str) -> None:
    if raw.lower() in ("q", "quit", "exit"):
        print("Zruseno.")
        sys.exit(0)


def _pick(prompt: str, options: list, default_idx: int = 0) -> str:
    print()
    print(prompt)
    for i, (_, desc) in enumerate(options, 1):
        marker = " *" if i - 1 == default_idx else "  "
        print(f"  {marker}{i}) {desc}")
    print("  (q = zrusit)")
    while True:
        try:
            raw = input(f"Vyber (1-{len(options)}, Enter = [{default_idx + 1}]): ").strip()
        except EOFError:
            return options[default_idx][0]
        _quit(raw)
        if raw == "":
            return options[default_idx][0]
        try:
            idx = int(raw) - 1
        except ValueError:
            print(f"  ! Zadej cislo 1-{len(options)}")
            continue
        if 0 <= idx < len(options):
            return options[idx][0]
        print(f"  ! Neplatna volba, zadej 1-{len(options)}")


def _input_int(prompt: str, default: int, min_val: int = 0, max_val: int = 100) -> int:
    print()
    while True:
        try:
            raw = input(f"{prompt} (Enter = {default}, q = zrusit): ").strip()
        except EOFError:
            return default
        _quit(raw)
        if raw == "":
            return default
        try:
            value = int(raw)
            if min_val <= value <= max_val:
                return value
            print(f"  ! Mimo rozsah {min_val}-{max_val}")
        except ValueError:
            print("  ! Zadej cele cislo")


def _yesno(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]:" if default else " [y/N]:"
    try:
        raw = input(f"{prompt}{suffix} ").strip().lower()
    except EOFError:
        return default
    _quit(raw)
    if raw == "":
        return default
    return raw == "y"


def _pick_input_source() -> str:
    """Najdi videa a nech uzivatele vybrat cislem."""
    videos = discover_videos(_PROJECT_ROOT)
    options: list[tuple[str, str]] = [("0", "Webcam (kamera 0)")]
    for label, path in videos:
        options.append((path, f"{label}"))

    if len(options) == 1:
        print()
        print("  ! Zadna videa v tests/, input/ ani videos/.")
        print("    Vloz .mp4 do slozky input/ a spust znovu,")
        print("    nebo pouzij webcam.")
        print("  *1) Webcam (kamera 0)")
        print("  (q = zrusit)")
        while True:
            raw = input("Vyber (1, Enter = [1]): ").strip()
            _quit(raw)
            if raw in ("", "1"):
                return "0"
            print("  ! Zadej 1 pro webcam")

    default_idx = 1 if len(options) > 1 else 0
    print()
    print("1) Vyber vstup:")
    for i, (_, desc) in enumerate(options, 1):
        marker = " *" if i - 1 == default_idx else "  "
        print(f"  {marker}{i}) {desc}")
    print("  (q = zrusit)")
    while True:
        try:
            raw = input(f"Vyber (1-{len(options)}, Enter = [{default_idx + 1}]): ").strip()
        except EOFError:
            return options[default_idx][0]
        _quit(raw)
        if raw == "":
            return options[default_idx][0]
        try:
            idx = int(raw) - 1
        except ValueError:
            print(f"  ! Zadej cislo 1-{len(options)}")
            continue
        if 0 <= idx < len(options):
            return options[idx][0]
        print(f"  ! Neplatna volba, zadej 1-{len(options)}")


def _safe_stem(input_source: str) -> str:
    if input_source == "0":
        return "webcam"
    stem = os.path.splitext(os.path.basename(input_source))[0]
    safe = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return safe[:60] or "video"


def _auto_output_paths(input_source: str, save_video: bool, save_frames: bool) -> tuple[str, str]:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    stem = _safe_stem(input_source)
    out_video = os.path.join(_OUTPUT_DIR, f"{stem}_anonymized.mp4") if save_video else ""
    out_frames = os.path.join(_OUTPUT_DIR, f"{stem}_frames") if save_frames else ""
    return out_video, out_frames


def run_interactive() -> dict:
    _print_header("VIDEO ANONYMIZER")
    print("Menu: staci vybirat cisla. Vystupy se ukladaji automaticky do output/.")

    input_source = _pick_input_source()

    detector = _pick("2) Detektor:", DETECTORS, default_idx=2)

    tracker = _pick("3) Tracker:", TRACKERS, default_idx=0)

    anon_method = _pick("4) Anonymizace:", ANON_METHODS, default_idx=1)

    anon_strength = 15
    if anon_method in ("mosaic", "blur"):
        anon_strength = _input_int("5) Sila efektu (1=jemne, 50=silne)", default=15, min_val=1, max_val=50)

    save_video = _yesno("6) Ulozit anonymizovane video", default=True)
    save_frames = _yesno("7) Ulozit jednotlive snimky (JPG)", default=False)
    show_window = _yesno("8) Zobrazit okno v realnem case", default=not save_video)

    do_backfill = _yesno("9) Doplnt pozni detekce zpetne (backfill)", default=False)

    show_boxes = _yesno("10) Zobrazit bounding boxy ve vystupu", default=True)

    out_video, out_frames = _auto_output_paths(input_source, save_video, save_frames)

    _print_header("SOUHRN")
    print(f"  Vstup:       {input_source}")
    print(f"  Detektor:    {detector}")
    print(f"  Tracker:     {tracker}")
    if anon_method == "none":
        print("  Anonymizace: vypnuta")
    else:
        print(f"  Anonymizace: {anon_method} (sila={anon_strength})")
    if save_video:
        print(f"  Video out:   {out_video}")
    if save_frames:
        print(f"  Frames out:  {out_frames}")
    print(f"  Okno:        {'ano' if show_window else 'ne'}")
    print(f"  Backfill:    {'ano' if do_backfill else 'ne'}")
    print(f"  Bounding boxy: {'ano' if show_boxes else 'ne'}")
    print()

    if not _yesno("Spustit", default=True):
        print("Zruseno.")
        sys.exit(0)

    return {
        "input": input_source,
        "detector": detector,
        "tracker": tracker,
        "anon_method": anon_method,
        "anon_strength": anon_strength,
        "save_video": save_video,
        "out_video": out_video,
        "save_frames": save_frames,
        "out_frames": out_frames,
        "show_window": show_window,
        "no_display": not show_window,
        "no_boxes": not show_boxes,
        "backfill": do_backfill,
        "config": os.path.join(_PROJECT_ROOT, "configs", "config.yaml"),
        "save_all": save_frames,
    }
