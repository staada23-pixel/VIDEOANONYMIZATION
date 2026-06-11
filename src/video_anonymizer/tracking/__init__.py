"""Trackery pro sledování objektů mezi snímky.

Moduly:
  base_tracker.py   — abstraktní BaseTracker (ABC)
  kcf.py            — KCF tracker s Gaussian kernelem + template matching fallback
  byte_tracker.py   — ByteTracker: asociace detekcí přes 3 kola IOU matching
  structures.py     — Track a TrackState dataclasses
"""
