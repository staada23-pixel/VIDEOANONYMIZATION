"""Camera Motion Compensation — Farneback optical flow."""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotionCompensator:
    """
    Odhad celkového pohybu kamery mezi dvěma snímky pomocí mediánu
    optického toku (Farneback). Kompaktní, robustní, neřeší rotaci/zoom.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.prev_gray = None
        self.scale: float = float(cfg.get("cmc_scale", 0.4))
        self.pyramid_scale: float = float(cfg.get("cmc_pyramid_scale", 0.5))
        self.levels: int = int(cfg.get("cmc_levels", 3))
        self.winsize: int = int(cfg.get("cmc_winsize", 21))
        self.iterations: int = int(cfg.get("cmc_iterations", 3))
        self.poly_n: int = int(cfg.get("cmc_poly_n", 5))
        self.poly_sigma: float = float(cfg.get("cmc_poly_sigma", 1.2))

    def compute(self, frame_gray) -> tuple[float, float]:
        """Vrať (dx, dy) — odhadovaný posun kamery v pixelech (full-res)."""
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return 0.0, 0.0

        h = int(frame_gray.shape[0] * self.scale)
        w = int(frame_gray.shape[1] * self.scale)
        prev_small = cv2.resize(self.prev_gray, (w, h))
        curr_small = cv2.resize(frame_gray, (w, h))
        flow = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None,
            self.pyramid_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma, 0,
        )
        dx = float(np.median(flow[..., 0])) / self.scale
        dy = float(np.median(flow[..., 1])) / self.scale
        self.prev_gray = frame_gray
        return dx, dy
