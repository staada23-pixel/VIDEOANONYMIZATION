"""Anonymizační operace nad bounding boxy.

Podporované metody:
  - pixelate:  rozdělí ROI na malé dlaždice (mozaika)
  - gaussian:  gaussovský blur přes ROI
  - blackout:  plný barevný obdélník (defaultně černá)
  - none:      žádná změna (pouze vizualizace)

Funkce pracují inplace nad frame_bgr (numpy array, BGR).
"""
import cv2


METHODS = ("pixelate", "gaussian", "blackout", "none")


def pixelate(frame_bgr, bbox, block=18):
    """Rozdělí ROI na malé dlaždice (block x block px) — silná anonymizace."""
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return
    roi = frame_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)),
                       interpolation=cv2.INTER_LINEAR)
    frame_bgr[y1:y2, x1:x2] = cv2.resize(small, (rw, rh),
                                         interpolation=cv2.INTER_NEAREST)


def gaussian_blur(frame_bgr, bbox, ksize=(51, 51)):
    """Gaussovský blur přes ROI."""
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return
    frame_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(frame_bgr[y1:y2, x1:x2], ksize, 0)


def blackout(frame_bgr, bbox, color=(0, 0, 0)):
    """Naplň ROI plnou barvou (defaultně černá)."""
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return
    frame_bgr[y1:y2, x1:x2] = color


def expand_bbox(bbox, factor, frame_shape):
    """Rozšíří bbox o `factor` velikosti na každou stranu (bez přesahu rámce)."""
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * factor)),
            max(0, int(y1 - bh * factor)),
            min(w, int(x2 + bw * factor)),
            min(h, int(y2 + bh * factor)))


def anonymize_region(frame_bgr, bbox, method="pixelate",
                     block=18, ksize=(51, 51), color=(0, 0, 0),
                     expand=0.15):
    """Hlavní anonymizační entry-point.

    method ∈ {"pixelate", "gaussian", "blackout", "none"}
    """
    if method == "none":
        return
    exp = expand_bbox(bbox, expand, frame_bgr.shape)
    if method == "blur" or method == "gaussian":
        gaussian_blur(frame_bgr, exp, ksize=ksize)
    elif method == "blackout":
        blackout(frame_bgr, exp, color=color)
    else:
        pixelate(frame_bgr, exp, block=block)


def draw_bbox(frame_bgr, bbox, color=(0, 255, 0), thickness=2, label=None,
              font_scale=0.5):
    """Vykreslí bbox + volitelný label. Inplace."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                             font_scale, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - baseline - 4)),
                      (x1 + tw, y1), color, -1)
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - baseline - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
