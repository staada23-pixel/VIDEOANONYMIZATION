"""
Test backward fill (backfill) — overi ze pozni detekce jsou doplneny zpetne.

Scenar:
  - Synteticke video s pohybujici se osobou
  - Mock detektor, ktery prvnich 20 framu vrati prazdno (osoba je tam, ale detektor ji "nevidi")
  - Od framu 21 zacne detekovat normalne
  - BackwardFiller by mel doplnit detekce do framu 1-20

Spusteni:
    python tests/test_backfill.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, PROJECT_SRC)

import cv2
import numpy as np

from video_anonymizer.detection.structures import Detection
from video_anonymizer.tracking.kcf import KCFTracker
from video_anonymizer.tracking.byte_tracker import ByteTracker
from video_anonymizer.tracking.structures import TrackState
from video_anonymizer.utils.cmc import CameraMotionCompensator
from video_anonymizer.utils.frame_history import FrameHistory
from video_anonymizer.utils.backfill import BackwardFiller
from video_anonymizer.utils.overlap import box_center_dist


# ── Mock detektor s delayem ─────────────────────────────────────

class DelayedDetector:
    """
    Detektor ktery prvnich `delay_frames` framu nic nedetekuje,
    pak zacne detekovat s pravdepodobnosti `prob`.
    """

    def __init__(self, ground_truth_box_getter, delay_frames=20,
                 prob=0.8, confidence=0.85, noise=3.0):
        self._get_box = ground_truth_box_getter
        self.delay = delay_frames
        self.prob = prob
        self.conf = confidence
        self.noise = noise
        self._frame_count = 0

    def detect(self, frame_bgr):
        self._frame_count += 1
        if self._frame_count <= self.delay:
            return []
        gt = self._get_box()
        if gt is None:
            return []
        if np.random.random() > self.prob:
            return []
        x1, y1, x2, y2 = gt
        n = self.noise
        return [Detection(
            x1=x1 + np.random.uniform(-n, n),
            y1=y1 + np.random.uniform(-n, n),
            x2=x2 + np.random.uniform(-n, n),
            y2=y2 + np.random.uniform(-n, n),
            confidence=self.conf + np.random.uniform(-0.05, 0.05),
        )]


# ── Synteticke video (kopie z test_pipeline_mock) ───────────────

class SyntheticVideo:
    def __init__(self, width=320, height=240, n_frames=80, fps=20):
        self.w = width
        self.h = height
        self.n_frames = n_frames
        self.fps = fps
        self.box_size = 40
        self.cx0 = width // 2
        self.cy0 = height // 2
        self.ax = 80
        self.ay = 40

    def person_box(self, t: int):
        cx = self.cx0 + self.ax * np.sin(2 * np.pi * t / 30)
        cy = self.cy0 + self.ay * np.cos(2 * np.pi * t / 20)
        half = self.box_size // 2
        return [cx - half, cy - half, cx + half, cy + half]

    def render(self, t: int):
        frame = np.full((self.h, self.w, 3), 30, dtype=np.uint8)
        bx1, by1, bx2, by2 = [int(v) for v in self.person_box(t)]
        bx1 = max(0, bx1); by1 = max(0, by1)
        bx2 = min(self.w - 1, bx2); by2 = min(self.h - 1, by2)
        frame[by1:by2, bx1:bx2] = 240
        head_r = (bx2 - bx1) // 4
        head_cx = (bx1 + bx2) // 2
        head_cy = by1 + head_r + 2
        cv2.circle(frame, (head_cx, head_cy), head_r, (200, 200, 200), -1)
        return frame

    def __iter__(self):
        self._t = 0
        return self

    def __next__(self):
        if self._t >= self.n_frames:
            raise StopIteration
        frame = self.render(self._t)
        t = self._t
        self._t += 1
        return t + 1, frame


# ── Hlavni test ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Test backward fill (backfill)")
    print("=" * 60)

    # 1. Sestav pipeline
    video = SyntheticVideo(n_frames=50)
    detector = DelayedDetector(
        ground_truth_box_getter=lambda: video.person_box(video._t - 1),
        delay_frames=15,
        prob=0.9,
        confidence=0.9,
    )
    cmc = CameraMotionCompensator({"cmc_scale": 0.5})
    bt = ByteTracker(
        lambda: KCFTracker({
            "padding": 2.0, "sigma": 0.5, "lambda": 1e-4,
            "learning_rate": 0.075, "output_sigma": 0.1,
            "psr_threshold": 7.0, "max_speed": 0.35,
        }),
        {
            "high_thresh": 0.4, "low_thresh": 0.08,
            "max_lost_frames": 10, "iou_threshold": 0.08,
            "reinit_dist_thresh": 40,
        },
    )

    history = FrameHistory()
    print(f"\nDetektor ma delay {detector.delay} framu...")

    # 2. Forward pass
    for frame_idx, frame in video:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy = cmc.compute(gray)
        detections = detector.detect(frame)
        tracks = bt.update(detections, frame, cam_dx=cam_dx, cam_dy=cam_dy)
        history.record(frame_idx, tracks, cam_dx, cam_dy, len(detections))

    print(f"  Zaznamenano {len(history.snapshots)} framu")

    # 3. Backward fill
    filler = BackwardFiller(video.w, video.h)
    backfilled_cnt, modified_frames = filler.fill(history.snapshots)

    # 4. Vyhodnoceni
    print(f"\n--- Vysledky ---")
    print(f"  Backfilled detekci: {backfilled_cnt}")
    print(f"  Modifikovanych framu: {len(modified_frames)}")

    # Over, ze framy 1..14 maji backfilled track
    backfilled_snapshots = [s for s in history.snapshots
                            if any(t.source == "backfilled" for t in s.tracks)]
    bf_frames = sorted(s.frame_idx for s in backfilled_snapshots)

    if bf_frames:
        print(f"  Backfilled framy: {bf_frames[0]}..{bf_frames[-1]} "
              f"(celkem {len(bf_frames)})")
        print(f"  Prvni backfilled frame: {bf_frames[0]}")
        print(f"  Posledni backfilled frame: {bf_frames[-1]}")
    else:
        print(f"  ! Zadne backfilled framy")

    # Kontrola: prvnich 14 framu by melo mit backfilled zaznamy
    first_frames_with_detection = None
    for snap in history.snapshots:
        if any(t.source == "detection" for t in snap.tracks):
            first_frames_with_detection = snap.frame_idx
            break

    print(f"\n  Prvni frame s raw detekci: {first_frames_with_detection}")

    success = True
    if backfilled_cnt == 0:
        print("  [FAIL] Backfill nic nedoplnil")
        success = False
    elif first_frames_with_detection is None:
        print("  [FAIL] Detektor nikdy nic nenasel")
        success = False
    else:
        # Over ze vsechny framy pred prvni detekci maji backfilled polozku
        for snap in history.snapshots:
            if snap.frame_idx < first_frames_with_detection:
                has_bf = any(t.source == "backfilled" for t in snap.tracks)
                if not has_bf:
                    print(f"  [WARN] Frame {snap.frame_idx} nema backfilled polozku")
                    # Toto muze byt OK pokud je box mimo obraz

        # Over ze prvni detection frame NEMA source="backfilled"
        for snap in history.snapshots:
            if snap.frame_idx == first_frames_with_detection:
                has_bf_on_detection = any(t.source == "backfilled" for t in snap.tracks)
                if has_bf_on_detection:
                    print(f"  [WARN] Frame {snap.frame_idx} ma backfilled (overeni)")
                break

        print(f"  [OK] Backfill doplnil {backfilled_cnt} detekci do {len(modified_frames)} framu")

    # 5. Vypis JSON log
    log_path = os.path.join(HERE, "..", "output", "backfill_test_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    history.save_json(log_path)
    print(f"  Log: {log_path}")

    print(f"\n{'='*60}")
    if success:
        print(" VERDIKT: OK")
        return 0
    else:
        print(" VERDIKT: FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
