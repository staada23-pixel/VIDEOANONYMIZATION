"""Kalman filter-based bounding box tracker.

Jednoduchý 2D Kalman filter sledující (cx, cy, w, h, vx, vy) bboxu.
Slouží jako fallback / lightweight varianta k CSRT/KCF.
"""
import cv2
import numpy as np


class KalmanBoxTracker:
    _id_counter = 0

    def __init__(self, bbox_x1y1x2y2):
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter
        self.lost_frames = 0
        self.active = True
        self.hits = 1
        self.age = 0

        x1, y1, x2, y2 = bbox_x1y1x2y2
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0

        self._kf = self._build_kf()
        self._kf.statePost = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0],
                                      dtype=np.float32).reshape(8, 1)
        self._kf.statePre = self._kf.statePost.copy()
        self.bbox = (x1, y1, x2, y2)

    @staticmethod
    def _build_kf():
        kf = cv2.KalmanFilter(8, 4)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1.0
        kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            kf.measurementMatrix[i, i] = 1.0
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        kf.errorCovPre = np.eye(8, dtype=np.float32)
        return kf

    def predict(self):
        self.age += 1
        if self.lost_frames > 0:
            self.hits = 0
        self.lost_frames += 1
        state = self._kf.predict()
        cx, cy, w, h = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        w = max(1.0, w)
        h = max(1.0, h)
        self.bbox = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
        return self.bbox

    def update(self, bbox_x1y1x2y2):
        x1, y1, x2, y2 = bbox_x1y1x2y2
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        self._kf.correct(measurement)
        self.bbox = (x1, y1, x2, y2)
        self.lost_frames = 0
        self.hits += 1
        self.active = True

    def get_bbox(self):
        return self.bbox

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0
