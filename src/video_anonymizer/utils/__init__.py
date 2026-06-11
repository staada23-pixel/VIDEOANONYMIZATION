"""Pomocné utility pro anonymizaci a zpracování videa.

Moduly:
  anonymizer.py     — anonymizační metody (mosaic, blur, black, solid)
  cmc.py            — Camera Motion Compensation (Farneback optical flow)
  logging_utils.py  — formátování výpisu a statistik
  overlap.py        — IoU, box_center_dist, greedy_match
  frame_history.py  — per-frame záznam detekcí/trackingu pro forward pass log
  backfill.py       — zpětné doplnění pozdních detekcí (inverzní CMC)
"""
