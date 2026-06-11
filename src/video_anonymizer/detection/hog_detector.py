"""OpenCV HOG person detector — fallback kdyz neni LPM/HASP k dispozici.

Nepotrebuje hardware, funguje out-of-the-box. Detekuje cele postavy
(HOG + SVM), ne obličeje. Pro bodycam footage s lidmi v záběru je to
rozumne pouzitelne.
"""
from __future__ import annotations

import cv2
import numpy as np

from .structures import Detection


class HOGDetector:
    """
    Obálka nad OpenCV HOG + SVM person detector.
    Rozhrani kompatibilni s LPMWrapper — ma `detect(frame) -> list[Detection]`.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.confidence: float = float(cfg.get("confidence", 0.5))
        self.hit_threshold: float = float(cfg.get("hit_threshold", 0.0))
        self.win_stride: tuple = tuple(cfg.get("win_stride", [4, 4]))
        self.padding: tuple = tuple(cfg.get("padding", [8, 8]))
        self.scale: float = float(cfg.get("scale", 1.05))
        self.group_threshold: int = int(cfg.get("group_threshold", 1))

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame_bgr) -> list[Detection]:
        """Detekuj osoby ve snimku. Vrat list Detection objektu."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            return []
        rects, weights = self.hog.detectMultiScale(
            frame_bgr,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            useMeanshiftGrouping=False,
        )
        detections: list[Detection] = []
        for (x, y, w, h), w_val in zip(rects, weights):
            # HOG SVM weights nejsou pravdepodobnosti (typicky 0.5–2.0)
            # Normalizujeme na 0–1 skrze sigmoidu
            try:
                raw = float(w_val[0])
            except (TypeError, IndexError):
                raw = float(w_val)
            score = 1.0 / (1.0 + 2.7 ** (-raw))
            if score < self.confidence:
                continue
            detections.append(Detection(
                x1=float(x), y1=float(y),
                x2=float(x + w), y2=float(y + h),
                confidence=score,
            ))
        return detections

    def close(self) -> None:
        pass
