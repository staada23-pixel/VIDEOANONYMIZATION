"""Wrapper kolem OpenCV trackera (CSRT, MIL) pro jednotne rozhrani."""
from __future__ import annotations

import cv2

from .base_tracker import BaseTracker


_OPENCV_MAP = {
    "CSRT": cv2.TrackerCSRT_create,
    "MIL": cv2.TrackerMIL_create,
}


class OpenCVTracker(BaseTracker):
    """
    Obalka nad OpenCV tracker (CSRT, MIL).
    Implementuje BaseTracker rozhrani pro jednotne pouziti v pipeline.
    """

    def __init__(self, kind: str, config: dict | None = None):
        if kind.upper() not in _OPENCV_MAP:
            raise ValueError(f"Neznamy OpenCV tracker: {kind}. Moznosti: {list(_OPENCV_MAP.keys())}")
        self._kind = kind.upper()
        self._tracker = _OPENCV_MAP[self._kind]()
        self._alive: bool = False
        self._ok: bool = True
        self._box: list = [0, 0, 0, 0]
        self._psr: float = 99.0

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def is_ok(self) -> bool:
        return self._ok

    @property
    def psr(self) -> float:
        return self._psr

    @property
    def template_score(self) -> float:
        return self._psr / 100.0 if self._alive else 0.0

    def init(self, frame_bgr, box) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if w < 5 or h < 5:
            self._alive = False
            return
        try:
            ok = self._tracker.init(frame_bgr, (x1, y1, w, h))
            self._alive = ok
            self._box = [float(x1), float(y1), float(x2), float(y2)]
        except Exception:
            self._alive = False

    def update(self, frame_bgr) -> bool:
        if not self._alive:
            return False
        try:
            ok, bbox = self._tracker.update(frame_bgr)
        except Exception:
            self._alive = False
            return False
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            self._box = [float(x), float(y), float(x + w), float(y + h)]
            self._ok = True
        else:
            self._ok = False
        return True

    def get_box(self) -> list:
        return self._box

    def apply_camera_motion(self, dx: float, dy: float) -> None:
        self._box = [
            self._box[0] + dx, self._box[1] + dy,
            self._box[2] + dx, self._box[3] + dy,
        ]
