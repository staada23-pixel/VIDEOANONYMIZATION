"""Plny smoke test na bodycam videu.

Spusti cely pipeline: face DNN YuNet + KCF tracker + mosaic anonymizer,
uloz anonymizovane video do project/output/.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from video_anonymizer.io.video_reader import VideoReader
from video_anonymizer.detection.face_detector import FaceDetector
from video_anonymizer.tracking.byte_tracker import ByteTracker
from video_anonymizer.tracking.kcf import KCFTracker
from video_anonymizer.utils.cmc import CameraMotionCompensator
from video_anonymizer.utils.anonymizer import Anonymizer


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
VIDEO = os.path.join(_PROJECT_ROOT, "..", "Praxe 2026", "EYEDEA PROJECT",
                     "LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2",
                     "wrappers", "python",
                     "capture of publicly available bodycam footage.mp4")
OUT_DIR = os.path.join(_PROJECT_ROOT, "output")
OUT_VIDEO = os.path.join(OUT_DIR, "bodycam_anonymized.mp4")
OUT_SAMPLES = os.path.join(OUT_DIR, "bodycam_samples")
MAX_FRAMES = 600  # pro rychly smoke test (~30s @ 20fps)


def main():
    if not os.path.exists(VIDEO):
        print(f"CHYBA: video neexistuje: {VIDEO}")
        return 1

    os.makedirs(OUT_SAMPLES, exist_ok=True)
    print(f"Video: {VIDEO}")
    print(f"  size: {os.path.getsize(VIDEO)/1024/1024:.1f} MB")

    reader = VideoReader(VIDEO)
    print(f"  {reader.width}x{reader.height} @ {reader.fps:.1f} fps, {len(reader)} snimku")

    # Detektor
    detector = FaceDetector({"confidence": 0.3, "min_face_size": 30})
    print(f"Detektor: {type(detector).__name__}")

    # Tracker
    cmc_cfg = {"cmc_scale": 0.4, "cmc_pyramid_scale": 0.5, "cmc_levels": 3,
               "cmc_winsize": 21, "cmc_iterations": 3, "cmc_poly_n": 5, "cmc_poly_sigma": 1.2}
    cmc = CameraMotionCompensator(cmc_cfg)
    bt = ByteTracker(lambda: KCFTracker({}), {})

    # Anonymizer
    anon = Anonymizer(method="mosaic", strength=18)
    print(f"Anonymizer: {anon.method} (sila={anon.strength})")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(OUT_VIDEO, fourcc, reader.fps if reader.fps > 0 else 20.0,
                         (reader.width, reader.height))
    if not vw.isOpened():
        print(f"CHYBA: nelze otevrit video writer: {OUT_VIDEO}")
        return 1
    print(f"Output video: {OUT_VIDEO}")

    sample_frames = [10, 50, 150, 300, 500]  # snimky co ulozime jako JPG pro kontrolu

    t0 = time.time()
    total_faces = 0
    total_tracks = 0
    frame_idx = 0
    processed = 0

    try:
        for frame_idx, frame in reader:
            if frame_idx >= MAX_FRAMES:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cam_dx, cam_dy = cmc.compute(gray) if cmc else (0.0, 0.0)

            dets = detector.detect(frame)
            total_faces += len(dets)
            tracks = bt.update(dets, frame, cam_dx=cam_dx, cam_dy=cam_dy)
            total_tracks += len(tracks)

            # 1. Anonymizuj cisty snimek
            out = frame.copy()
            for t in tracks:
                anon.apply(out, t.box)

            # 2. Vykresli boxy
            for t in tracks:
                x1, y1, x2, y2 = [int(v) for v in t.box]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Info bar
            cv2.putText(out, f"frame {frame_idx} det={len(dets)} tr={len(tracks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            vw.write(out)

            # Uloz vzorky pro kontrolu
            if frame_idx in sample_frames:
                path = os.path.join(OUT_SAMPLES, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(path, out)
                print(f"  sample: {path} (dets={len(dets)}, tr={len(tracks)})")

            processed += 1
            if processed % 100 == 0:
                print(f"  ... {processed} snimku zpracovano")
    finally:
        vw.release()
        reader.release()
        detector.close()

    elapsed = time.time() - t0
    fps = processed / elapsed if elapsed > 0 else 0
    print(f"\nHotovo: {processed} snimku za {elapsed:.1f}s = {fps:.1f} fps")
    print(f"  detekci celkem: {total_faces} (prumer {total_faces/processed:.2f}/snimek)")
    print(f"  tracku celkem:  {total_tracks} (prumer {total_tracks/processed:.2f}/snimek)")
    print(f"  video out: {OUT_VIDEO}")
    print(f"  samples:   {OUT_SAMPLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
