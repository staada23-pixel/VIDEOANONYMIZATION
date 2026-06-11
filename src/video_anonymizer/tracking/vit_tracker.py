"""ViTTrack tracker — Vision Transformer based (SOTA, OpenCV 4.8+)."""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

from .base_tracker import BaseTracker

_VITTRACK_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
)
_VITTRACK_FILENAME = "object_tracking_vittrack_2023sep.onnx"


def _project_models_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _download_vittrack(dest: str) -> bool:
    try:
        print(f"[vit_tracker] Stahuji ViTTrack model z OpenCV Zoo...")
        urllib.request.urlretrieve(_VITTRACK_URL, dest)
        print(f"[vit_tracker] OK: {dest}  ({os.path.getsize(dest)/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"[vit_tracker] Stazeni selhalo: {e}")
        return False


class ViTTracker(BaseTracker):
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        models_dir = _project_models_dir()
        model_path = os.path.join(models_dir, _VITTRACK_FILENAME)

        if not os.path.isfile(model_path):
            if not _download_vittrack(model_path):
                raise RuntimeError("Nepodarilo se stahnout ViTTrack model")

        self._params = cv2.TrackerVit_Params()
        self._params.net = model_path
        self._tracker = cv2.TrackerVit_create(self._params)

        self._box: list | None = None
        self._alive: bool = False
        self._ok: bool = True
        self._psr: float = 99.0
        self._tmpl_score: float = 0.0

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
        return self._tmpl_score

    def init(self, frame_bgr, box) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bw < 10 or bh < 10:
            self._alive = False
            return
        bbox = (x1, y1, bw, bh)
        self._tracker.init(frame_bgr, bbox)
        self._box = [float(v) for v in box]
        self._alive = True
        self._ok = True

    def update(self, frame_bgr) -> bool:
        if not self._alive:
            return False
        ok, bbox = self._tracker.update(frame_bgr)
        if not ok:
            self._ok = False
            return True
        x, y, w, h = [int(v) for v in bbox]
        self._box = [float(x), float(y), float(x + w), float(y + h)]
        score = self._tracker.getTrackingScore()
        self._psr = float(score * 100)
        self._ok = score > 0.2
        return True

    def get_box(self) -> list:
        return self._box if self._box is not None else [0, 0, 0, 0]

    def apply_camera_motion(self, dx: float, dy: float) -> None:
        if self._alive and self._box is not None:
            self._box = [
                self._box[0] + dx, self._box[1] + dy,
                self._box[2] + dx, self._box[3] + dy,
            ]

    def refresh_template(self, frame_bgr) -> None:
        pass
