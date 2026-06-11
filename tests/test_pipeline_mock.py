"""
Test pipeline bez LPM hardwaroveho klice.

Vytvori synteticke video s pohybujici se "osobou" (bily ctverec),
mock detektor ktery vraci detekce na spravnem miste, a overi ze
KCF tracker + ByteTracker funguji spravne.

Spusteni:
    python tests/test_pipeline_mock.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# Pridej src a LPM wrappers do PYTHONPATH
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, PROJECT_SRC)

import cv2
import numpy as np

from video_anonymizer.detection.structures import Detection
from video_anonymizer.detection.hog_detector import HOGDetector
from video_anonymizer.tracking.kcf import KCFTracker
from video_anonymizer.tracking.byte_tracker import ByteTracker
from video_anonymizer.tracking.structures import TrackState
from video_anonymizer.utils.cmc import CameraMotionCompensator
from video_anonymizer.io.frame_writer import FrameWriter


# ── Mock detektor ───────────────────────────────────────────────

class MockLPMDetector:
    """
    Falesny detektor ktery "vidi" pohybujici se ctverec v obrazu.
    Vraci Detection jen nekdy (simuluje neperfektni LPM).
    """

    def __init__(self, ground_truth_box_getter, detect_probability=0.6,
                 confidence=0.85, noise=3.0):
        self._get_box = ground_truth_box_getter
        self.prob = detect_probability
        self.conf = confidence
        self.noise = noise

    def detect(self, frame_bgr):
        gt = self._get_box()
        if gt is None:
            return []
        if np.random.random() > self.prob:
            return []  # detektor obcas minne
        x1, y1, x2, y2 = gt
        # Pridej maly sum aby to bylo realisticke
        n = self.noise
        return [Detection(
            x1=x1 + np.random.uniform(-n, n),
            y1=y1 + np.random.uniform(-n, n),
            x2=x2 + np.random.uniform(-n, n),
            y2=y2 + np.random.uniform(-n, n),
            confidence=self.conf + np.random.uniform(-0.05, 0.05),
        )]


# ── Synteticke video ────────────────────────────────────────────

class SyntheticVideo:
    """
    Generator snimku s pohybujicim se "osobou" (bily ctverec) na tmavem pozadi.
    Simuluje kameru ktera se pomalu pohybuje (CMC).
    """

    def __init__(self, width=320, height=240, n_frames=80, fps=20):
        self.w = width
        self.h = height
        self.n_frames = n_frames
        self.fps = fps
        self.box_size = 40
        # Pohyb osoby: sinusoida v x a y
        self.cx0 = width // 2
        self.cy0 = height // 2
        self.ax = 80
        self.ay = 40

    def person_box(self, t: int):
        """Ground-truth bounding box osoby v case t."""
        cx = self.cx0 + self.ax * np.sin(2 * np.pi * t / 30)
        cy = self.cy0 + self.ay * np.cos(2 * np.pi * t / 20)
        half = self.box_size // 2
        return [cx - half, cy - half, cx + half, cy + half]

    def camera_offset(self, t: int):
        """Pomalý posun kamery (aby CMC melo co delat)."""
        return (2.0 * t, 1.0 * t)  # (dx, dy) za snimek

    def render(self, t: int):
        """Vygeneruj snimek s 'osobou' a 'pozadim'."""
        frame = np.full((self.h, self.w, 3), 30, dtype=np.uint8)
        # Pridat nejaky texturovany background
        rng = np.random.default_rng(t)
        bg_noise = rng.integers(20, 50, size=(self.h, self.w), dtype=np.uint8)
        frame[..., 0] = bg_noise
        frame[..., 1] = bg_noise
        frame[..., 2] = bg_noise

        # Osoba - bily ctverec
        bx1, by1, bx2, by2 = [int(v) for v in self.person_box(t)]
        bx1 = max(0, bx1); by1 = max(0, by1)
        bx2 = min(self.w - 1, bx2); by2 = min(self.h - 1, by2)
        frame[by1:by2, bx1:bx2] = 240
        # Hlava (v horni casti)
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
    print(" Video Anonymizer — pipeline test s mock LPM")
    print("=" * 60)

    # Vystup do project/output/ (vedle src/, ne na Desktop)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "video_anonymizer_test")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Vystup: {out_dir}")

    # 1. Sestav pipeline
    video = SyntheticVideo(n_frames=60)
    detector = MockLPMDetector(
        ground_truth_box_getter=lambda: video.person_box(video._t - 1),
        detect_probability=0.7,
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
            "max_lost_frames": 6, "iou_threshold": 0.08,
            "reinit_dist_thresh": 40,
        },
    )
    writer = FrameWriter(
        output_dir=os.path.join(out_dir, "frames"),
        save_only_with_detections=False,  # chceme vsechny snimky
        save_every_n=5,
    )

    # 2. Spust pipeline
    print("\nSpoustim pipeline na 60 syntetickych snimcich...")
    start = time.time()
    stats = {
        "frames": 0, "lpm_hits": 0, "tracks_created": 0,
        "active_frames": 0, "lost_frames": 0, "saved_frames": 0,
    }
    last_track_box = None
    last_person_box = None
    iou_scores = []

    for frame_idx, frame in video:
        stats["frames"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy = cmc.compute(gray)

        detections = detector.detect(frame)
        if detections:
            stats["lpm_hits"] += 1

        prev_track_count = len(bt.tracks)
        tracks = bt.update(detections, frame, cam_dx=cam_dx, cam_dy=cam_dy)
        if len(bt.tracks) > prev_track_count:
            stats["tracks_created"] += len(bt.tracks) - prev_track_count

        # Anotuj snimek
        annotated = frame.copy()
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr.box]
            if tr.state == TrackState.ACTIVE:
                color = (0, 0, 255)  # cervena = LPM confirm
                label = f"ID{tr.id} ACT"
            elif tr.state == TrackState.LOST:
                color = (140, 40, 140)  # fialova = KCF/template
                label = f"ID{tr.id} TM:{tr.kcf_template_score:.2f}"
            else:
                color = (0, 200, 255)  # zluta
                label = f"ID{tr.id} LOW"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Porovnej s ground truth
            gt = video.person_box(frame_idx - 1)
            from video_anonymizer.utils.overlap import iou
            iou_val = iou(tr.box, gt)
            iou_scores.append(iou_val)
            last_track_box = tr.box
            last_person_box = gt

            if tr.state == TrackState.ACTIVE:
                stats["active_frames"] += 1
            else:
                stats["lost_frames"] += 1

        # Info nahore
        info = f"F:{frame_idx:03d} tracks:{len(tracks)} LPM:{len(detections)} CMC:{cam_dx:+.1f},{cam_dy:+.1f}"
        cv2.putText(annotated, info, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if writer.write(frame_idx, annotated, has_detections=len(tracks) > 0):
            stats["saved_frames"] += 1

    elapsed = time.time() - start

    # 3. Statistiky
    print(f"\n--- Statistiky ---")
    print(f"  Snimku zpracovano:  {stats['frames']}")
    print(f"  LPM detekci:        {stats['lpm_hits']} ({100*stats['lpm_hits']/stats['frames']:.0f}%)")
    print(f"  Tracku vytvoreno:   {stats['tracks_created']}")
    print(f"  ACTIVE snimku:      {stats['active_frames']}")
    print(f"  LOST/LOW snimku:    {stats['lost_frames']}")
    print(f"  Snimku ulozeno:     {stats['saved_frames']}")
    print(f"  Cas:                {elapsed:.2f} s ({stats['frames']/elapsed:.1f} FPS)")
    print(f"  Ulozeno do:         {os.path.join(out_dir, 'frames')}")

    if iou_scores:
        iou_arr = np.array(iou_scores)
        print(f"\n--- Kvalita trackingu (IoU vuci ground truth) ---")
        print(f"  Prumer:  {iou_arr.mean():.3f}")
        print(f"  Median:  {np.median(iou_arr):.3f}")
        print(f"  Min:     {iou_arr.min():.3f}")
        print(f"  Max:     {iou_arr.max():.3f}")
        print(f"  > 0.5:   {(iou_arr > 0.5).sum()}/{len(iou_arr)} "
              f"({100*(iou_arr > 0.5).sum()/len(iou_arr):.0f}%)")

    # 4. Verdikt
    print(f"\n--- Verdikt ---")
    if iou_scores and np.median(iou_scores) > 0.5:
        print("  [OK] Tracker sleduje osobu velmi dobre (median IoU > 0.5)")
    elif iou_scores and np.median(iou_scores) > 0.2:
        print("  [WARN] Tracker sleduje osobu jen priblizne (median IoU 0.2-0.5)")
    else:
        print("  [FAIL] Tracker ztraci osobu")
        return 1

    if stats["tracks_created"] <= 2:
        print(f"  [OK] Vytvoreno malo tracku ({stats['tracks_created']}) - ID je stabilni")
    else:
        print(f"  [WARN] Vytvoreno {stats['tracks_created']} tracku - prilis mnoho reinicializaci")

    print(f"\nOtevri par snimku z: {os.path.join(out_dir, 'frames')}")
    print("Pokud v nich vidis fialovy box na osobe, KCF+template fallback funguje!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
