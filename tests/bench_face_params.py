"""Benchmark ruznych detection parametru na bodycam videu."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from video_anonymizer.detection.face_detector import FaceDetector

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
VIDEO = os.path.join(_ROOT, "..", "Praxe 2026", "EYEDEA PROJECT",
                     "LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2",
                     "wrappers", "python",
                     "capture of publicly available bodycam footage.mp4")
N = 200

cap = cv2.VideoCapture(VIDEO)
print(f"  conf_thr  min_size  detections/{N}")
for cfgs in [
    {"confidence": 0.6, "min_face_size": 60},
    {"confidence": 0.4, "min_face_size": 40},
    {"confidence": 0.3, "min_face_size": 30},
    {"confidence": 0.2, "min_face_size": 20},
]:
    det = FaceDetector(cfgs)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = 0
    n = 0
    for i in range(N):
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        total += len(det.detect(f))
    det.close()
    print(f"  {cfgs['confidence']:4.2f}      {cfgs['min_face_size']:3d}       {total}/{n} = {total/n:.2f}/f")
cap.release()
