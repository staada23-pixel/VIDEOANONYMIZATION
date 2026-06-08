"""Jednoduchý IoU-based tracker.

Přiřazuje detekce k existujícím trackům podle IoU překryvu.
Nové tracky vznikají z nepřiřazených detekcí, ztracené tracky se
evidují po dobu `max_lost` framů, pak se ruší.
"""
from typing import List
from .track import Track
from ..utils.overlap_fn import iou


class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._id_counter = 0
        self._tracks: List[Track] = []

    def update(self, detections):
        """detections: list of (x1, y1, x2, y2, conf) or (x1, y1, x2, y2, conf, label)."""
        for t in self._tracks:
            t.lost_frames += 1
            t.age += 1
            t.active = False

        unmatched = list(range(len(self._tracks)))
        used_dets = set()

        for di, det in enumerate(detections):
            bbox = det[:4]
            conf = float(det[4]) if len(det) > 4 else 0.0
            label = det[5] if len(det) > 5 else ""
            best_iou, best_ti = 0.0, -1
            for ti in unmatched:
                iou_val = iou(self._tracks[ti].bbox, bbox)
                if iou_val > best_iou:
                    best_iou, best_ti = iou_val, ti
            if best_ti >= 0 and best_iou >= self.iou_threshold:
                t = self._tracks[best_ti]
                t.bbox = bbox
                t.confidence = conf
                t.lost_frames = 0
                t.hits += 1
                t.active = True
                t.label = label
                unmatched.remove(best_ti)
                used_dets.add(di)
            else:
                self._id_counter += 1
                self._tracks.append(Track(
                    id=self._id_counter,
                    bbox=bbox,
                    confidence=conf,
                    label=label,
                ))

        self._tracks = [t for t in self._tracks if t.lost_frames <= self.max_lost]
        return [t for t in self._tracks if t.active]

    @property
    def tracks(self):
        return list(self._tracks)
