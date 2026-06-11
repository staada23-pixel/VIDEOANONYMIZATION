"""Anonymizacni metody pro detekovane obliceje/objekty.

Dostupne metody:
  - mosaic  — pixelate (zmensi + zvetsi s nearest-neighbor)
  - blur    — Gaussovo rozmazani
  - black   — cerny obdelnik
  - solid   — jednobarevny obdelnik
"""
from __future__ import annotations

import cv2
import numpy as np


METHODS = ("none", "mosaic", "blur", "black", "solid")


class Anonymizer:
    """Aplikuje anonymizaci na ROI ve snimku."""

    def __init__(self, method: str = "mosaic", strength: int = 15,
                 color: tuple[int, int, int] = (0, 0, 0)):
        if method not in METHODS:
            raise ValueError(f"method musí být jedno z {METHODS}, ne {method!r}")
        self.method = method
        self.strength = max(1, int(strength))
        self.color = tuple(int(c) for c in color)

    def apply(self, frame_bgr, box) -> "np.ndarray":
        """
        Anonymizuj ROI v `frame_bgr` podle `box` (x1,y1,x2,y2).
        Vraci modifikovany frame (in-place i return).
        """
        if self.method == "none":
            return frame_bgr

        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        if not box or len(box) != 4:
            return frame_bgr

        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except (TypeError, ValueError):
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        # Crop na snimek
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return frame_bgr

        roi_w = x2 - x1
        roi_h = y2 - y1

        if self.method == "mosaic":
            scale = 1.0 + self.strength * 0.25
            cells = max(3, int(round(min(roi_w, roi_h) / scale)))
            small = cv2.resize(frame_bgr[y1:y2, x1:x2], (cells, cells),
                               interpolation=cv2.INTER_LINEAR)
            frame_bgr[y1:y2, x1:x2] = cv2.resize(
                small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
            )
        elif self.method == "blur":
            k = max(3, self.strength * 2 + 1)
            # k musi byt liche
            if k % 2 == 0:
                k += 1
            k = min(k, min(roi_w, roi_h) * 2 - 1)
            if k >= 3:
                frame_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(
                    frame_bgr[y1:y2, x1:x2], (k, k), 0
                )
        elif self.method == "black":
            frame_bgr[y1:y2, x1:x2] = 0
        elif self.method == "solid":
            frame_bgr[y1:y2, x1:x2] = self.color

        return frame_bgr
