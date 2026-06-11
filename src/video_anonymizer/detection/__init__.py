"""Detektory objektů/osob ve videu.

Moduly:
  face_detector.py  — YuNet DNN (obličeje) + Haar cascade fallback
  hog_detector.py   — HOG + SVM (celé postavy)
  lpm_wrapper.py    — Eyedea LPM SDK wrapper (vyžaduje HASP)
  structures.py     — Detection dataclass (x1, y1, x2, y2, confidence)
"""
