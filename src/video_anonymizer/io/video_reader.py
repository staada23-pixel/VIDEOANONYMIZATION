"""Video/image reading — oddělená I/O vrstva pro čtení snímků.

Podporované vstupy:
  - video soubor (.mp4/.avi/.mov/.mkv)
  - jeden obrázek (.jpg/.png/.bmp)
  - adresář s obrázky (sekvence)
  - webcam index (int)

Samonosný modul: žádná závislost na LPM/trackeru/anonymizeru.
"""
from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def detect_input_type(path):
    """Vrátí 'video' | 'image' | 'image_dir' | 'webcam' | 'unknown'."""
    if path is None:
        return "webcam"
    p = Path(path)
    if not p.exists():
        return "unknown"
    if p.is_dir():
        return "image_dir"
    ext = p.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return "unknown"


def open_video(path):
    """Zpětná kompatibilita: otevře video nebo webcam."""
    if path is None:
        return cv2.VideoCapture(0)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    return cv2.VideoCapture(str(p))


def open_input(path, input_type=None):
    """Dispatcher: vrátí (input_type, source_object).

    Pro 'image' vrací numpy array jako statický zdroj (source = ndarray).
    Pro 'image_dir' vrací (sorted_paths, first_image).
    Pro 'video'/'webcam' vrací cv2.VideoCapture.
    """
    if input_type is None:
        input_type = detect_input_type(path)

    if input_type == "video" or input_type == "webcam":
        cap = cv2.VideoCapture(0 if path is None else str(path))
        return input_type, cap

    if input_type == "image":
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return input_type, img

    if input_type == "image_dir":
        p = Path(path)
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        if not files:
            raise FileNotFoundError(f"No images in directory: {path}")
        return input_type, files

    raise ValueError(f"Unknown input type for path: {path}")


def iter_frames(source, input_type):
    """Generator přes BGR snímky (numpy ndarray). Ukončí se při vyčerpání.

    Pro input_type='image' vrátí jeden frame a skončí.
    Pro input_type='image_dir' čte soubory jeden po druhém.
    Pro input_type='video'/'webcam' čte z VideoCapture.
    """
    if input_type == "image":
        if isinstance(source, np.ndarray):
            yield source
        return

    if input_type == "image_dir":
        for path in source:
            img = cv2.imread(str(path))
            if img is not None:
                yield img
        return

    if input_type in ("video", "webcam"):
        while True:
            ok, frame = source.read()
            if not ok:
                return
            yield frame


def video_meta(cap):
    """Vrátí dict s fps, width, height, total. Pro cap=VideoCapture."""
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
