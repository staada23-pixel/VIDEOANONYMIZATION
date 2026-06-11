"""
Test s SKUTECNYM HOG detektorem (bez LPM, bez HASP).

Vytvori synteticke video s osobou (lidska postava), necha OpenCV
HOG ji najit a KCF trackovat. Vystup jde na Desktop.

Spusteni:
    python tests/test_hog_real.py
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

from video_anonymizer.detection.hog_detector import HOGDetector
from video_anonymizer.tracking.kcf import KCFTracker
from video_anonymizer.tracking.byte_tracker import ByteTracker
from video_anonymizer.tracking.structures import TrackState
from video_anonymizer.io.frame_writer import FrameWriter
from video_anonymizer.utils.overlap import iou


# ── Syntetická "osoba" — lidská postava z jednoduchých tvarů ─────

class PersonFigure:
    """Bílá postava s hlavou, tělem, nohama na tmavém pozadí."""

    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.cx0 = w // 2
        self.cy0 = h // 2
        self.ax = 100
        self.ay = 50
        self.scale = 1.5  # větší než default 40px box, aby HOG zachytil

    def position(self, t):
        cx = self.cx0 + self.ax * np.sin(2 * np.pi * t / 40)
        cy = self.cy0 + self.ay * np.cos(2 * np.pi * t / 30)
        return cx, cy

    def box(self, t):
        cx, cy = self.position(t)
        half_w = int(30 * self.scale)
        half_h = int(70 * self.scale)
        return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

    def render(self, t):
        frame = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        cx, cy = self.position(t)
        # Hlava
        head_r = int(12 * self.scale)
        cv2.circle(frame, (int(cx), int(cy - 50 * self.scale)), head_r, (230, 230, 230), -1)
        # Tělo (obdelník)
        body_w = int(25 * self.scale)
        body_h = int(50 * self.scale)
        x1 = int(cx - body_w // 2)
        y1 = int(cy - 25 * self.scale)
        x2 = x1 + body_w
        y2 = y1 + body_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), -1)
        # Nohy
        leg_w = int(8 * self.scale)
        leg_h = int(35 * self.scale)
        for off in (-10, 10):
            lx = int(cx + off * self.scale)
            ly = int(cy + 25 * self.scale)
            cv2.rectangle(frame, (lx - leg_w // 2, ly), (lx + leg_w // 2, ly + leg_h), (210, 210, 210), -1)
        return frame

    def __iter__(self):
        self._t = 0
        return self

    def __next__(self):
        if self._t >= 60:
            raise StopIteration
        f = self.render(self._t)
        t = self._t
        self._t += 1
        return t + 1, f


def main():
    print("=" * 60)
    print(" Video Anonymizer — REAL HOG test (bez HASP)")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "video_anonymizer_hog_test")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Vystup: {out_dir}")

    # Skutecne objekty
    person = PersonFigure(640, 480)
    detector = HOGDetector({"confidence": 0.3})  # nizsi prah pro HOG
    bt = ByteTracker(
        lambda: KCFTracker({
            "padding": 2.5, "sigma": 0.5, "lambda": 1e-4,
            "learning_rate": 0.075, "output_sigma": 0.1,
            "psr_threshold": 7.0, "max_speed": 0.35,
            "tm_search_mult": 3.0, "tm_min_score": 0.30,
        }),
        {
            "high_thresh": 0.3, "low_thresh": 0.1,
            "max_lost_frames": 8, "iou_threshold": 0.08,
            "reinit_dist_thresh": 60,
        },
    )
    writer = FrameWriter(
        output_dir=os.path.join(out_dir, "frames"),
        save_only_with_detections=False,
        save_every_n=3,
    )

    print("\nSpoustim 60 snimku se skutecnym HOG detektorem...\n")
    start = time.time()
    stats = {"frames": 0, "detections": 0, "tracks_created": 0,
             "active": 0, "lost": 0, "saved": 0, "ious": []}

    for frame_idx, frame in person:
        stats["frames"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy = 0.0, 0.0  # zadny CMC v tomto testu

        detections = detector.detect(frame)
        stats["detections"] += len(detections)

        prev = len(bt.tracks)
        tracks = bt.update(detections, frame, cam_dx=cam_dx, cam_dy=cam_dy)
        stats["tracks_created"] += len(bt.tracks) - prev

        # Anotuj
        annotated = frame.copy()
        gt = person.box(frame_idx - 1)
        cv2.rectangle(annotated, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])),
                      (0, 255, 0), 1)  # ground truth = zelený

        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            if tr.state == TrackState.ACTIVE:
                color = (0, 0, 255)
                label = f"ID{tr.id} ACT"
                stats["active"] += 1
            else:
                color = (140, 40, 140)
                label = f"ID{tr.id} TM:{tr.kcf_template_score:.2f}"
                stats["lost"] += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if tr.box and gt:
                stats["ious"].append(iou(tr.box, gt))

        # Info
        info = f"F:{frame_idx} HOG:{len(detections)} tracks:{len(tracks)}"
        cv2.putText(annotated, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if writer.write(frame_idx, annotated, has_detections=len(tracks) > 0):
            stats["saved"] += 1

    elapsed = time.time() - start

    # Statistiky
    print("--- Statistiky ---")
    print(f"  Snimku zpracovano:  {stats['frames']}")
    print(f"  HOG detekci:        {stats['detections']} ({stats['detections']/stats['frames']:.1f}/frame)")
    print(f"  Tracku vytvoreno:   {stats['tracks_created']}")
    print(f"  ACTIVE snimku:      {stats['active']}")
    print(f"  LOST snimku:        {stats['lost']}")
    print(f"  Snimku ulozeno:     {stats['saved']}")
    print(f"  Cas:                {elapsed:.2f} s ({stats['frames']/elapsed:.1f} FPS)")
    print(f"  Vystup:             {os.path.join(out_dir, 'frames')}")

    if stats["ious"]:
        iou_arr = np.array(stats["ious"])
        print(f"\n--- Kvalita trackingu (IoU vuci ground truth) ---")
        print(f"  Prumer: {iou_arr.mean():.3f}")
        print(f"  Median: {np.median(iou_arr):.3f}")
        print(f"  > 0.3:  {(iou_arr > 0.3).sum()}/{len(iou_arr)}")

    print(f"\n--- Verdikt ---")
    if stats["detections"] > 0:
        print(f"  [OK] HOG detektor funguje — nasel {stats['detections']} osob")
    else:
        print(f"  [WARN] HOG nenasel zadne osoby (zkus snizit confidence)")
        return 1

    if stats["tracks_created"] > 0 and stats["active"] + stats["lost"] > 0:
        print(f"  [OK] Tracker dostal detekce a sledoval osobu")
    else:
        print(f"  [WARN] Tracker nedostal detekce")

    print(f"\nOtevri: explorer \"{os.path.join(out_dir, 'frames')}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
