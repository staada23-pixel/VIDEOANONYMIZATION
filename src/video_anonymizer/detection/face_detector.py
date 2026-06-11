"""Face detector — OpenCV DNN (YuNet) + Haar cascade fallback.

Nepotrebuje HASP, funguje out-of-the-box. Specialne navrzeno pro
bodycam zaznamy s lidskymi oblicaji. Detekuje OBLIČEJE, ne cele postavy.
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np

import numpy as np

from .structures import Detection


# OpenCV Zoo model — malý (~350 KB), rychlý, přesný
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def _project_models_dir() -> str:
    """Adresar models/ v projektu (project/models/)."""
    # this file: project/src/video_anonymizer/detection/face_detector.py
    #   here  = .../project/src/video_anonymizer/detection/
    #   ../.. = .../project/src/
    #   ../../.. = .../project/
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _download_yunet(dest: str) -> bool:
    """Stahni YuNet ONNX model. Vrat True pri uspechu."""
    try:
        print(f"[face_detector] Stahuji YuNet model z OpenCV Zoo...")
        urllib.request.urlretrieve(_YUNET_URL, dest)
        print(f"[face_detector] OK: {dest}  ({os.path.getsize(dest)/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"[face_detector] Stazeni selhalo: {e}")
        return False


class FaceDetector:
    """
    Face detector. Prioritne YuNet (DNN), fallback Haar cascade.
    Rozhrani kompatibilni s LPMWrapper — `detect(frame) -> list[Detection]`.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.confidence: float = float(cfg.get("confidence", 0.55))
        self.min_face_size: int = int(cfg.get("min_face_size", 30))
        self.max_aspect_ratio: float = float(cfg.get("max_aspect_ratio", 1.5))
        self.nms_threshold: float = float(cfg.get("nms_threshold", 0.3))
        self.edge_margin: float = float(cfg.get("edge_margin", 0.02))

        models_dir = _project_models_dir()
        self._yunet_path = os.path.join(models_dir, _YUNET_FILENAME)

        self._yunet = None
        if os.path.exists(self._yunet_path) or _download_yunet(self._yunet_path):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    self._yunet_path,
                    "",
                    (320, 320),
                    score_threshold=self.confidence,
                    nms_threshold=0.1,
                    top_k=50,
                )
                self.method = "yunet"
                print(f"[face_detector] Pouzivam YuNet DNN (model: {_YUNET_FILENAME})")
            except Exception as e:
                print(f"[face_detector] YuNet selhal: {e}")
                self._yunet = None

        if self._yunet is None:
            haar_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            self._haar = cv2.CascadeClassifier(haar_path)
            if self._haar.empty():
                raise RuntimeError("Ani YuNet ani Haar cascade nejsou dostupne!")
            self.method = "haar"
            print("[face_detector] Pouzivam Haar cascade (fallback)")

    def detect(self, frame_bgr) -> list[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            return []
        if self._yunet is not None:
            return self._detect_yunet(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _apply_nms(self, detections: list[Detection]) -> list[Detection]:
        if len(detections) <= 1:
            return detections
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept = []
        for d in dets:
            dup = False
            for k in kept:
                if self._iou(d.box, k.box) > self.nms_threshold:
                    dup = True
                    break
            if not dup:
                kept.append(d)
        return kept

    @staticmethod
    def _iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    def _detect_yunet(self, frame_bgr) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(frame_bgr)
        raw: list[Detection] = []
        if faces is None:
            return raw
        mx = int(w * self.edge_margin)
        my = int(h * self.edge_margin)
        for f in faces:
            x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[14]
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + fw), float(y + fh)
            if not self._is_valid(x1, y1, x2, y2, float(score)):
                continue
            if x1 < mx or y1 < my or x2 > w - mx or y2 > h - my:
                continue
            landmarks = [(float(f[4 + i * 2]), float(f[5 + i * 2])) for i in range(5)]
            raw.append(Detection(x1, y1, x2, y2, float(score), landmarks=landmarks))
        return self._apply_nms(raw)

    def _detect_haar(self, frame_bgr) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        detections: list[Detection] = []
        for (x, y, fw, fh) in faces:
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + fw), float(y + fh)
            if not self._is_valid(x1, y1, x2, y2, 0.5):
                continue
            detections.append(Detection(x1, y1, x2, y2, 0.5))
        return detections

    def _is_valid(self, x1: float, y1: float, x2: float, y2: float,
                  score: float) -> bool:
        if score < self.confidence:
            return False
        w = x2 - x1
        h = y2 - y1
        if w < self.min_face_size or h < self.min_face_size:
            return False
        ratio = max(w / h, h / w)
        if ratio > self.max_aspect_ratio:
            return False
        return True

    def close(self) -> None:
        pass
