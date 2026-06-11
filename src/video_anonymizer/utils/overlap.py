"""Box overlap, distance a greedy matching utility."""
from __future__ import annotations

import numpy as np


def iou(boxA, boxB) -> float:
    """Intersection over Union dvou boxů [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0.0


def box_center_dist(boxA, boxB) -> float:
    """Euklidovská vzdálenost středů dvou boxů."""
    cax = (boxA[0] + boxA[2]) / 2
    cay = (boxA[1] + boxA[3]) / 2
    cbx = (boxB[0] + boxB[2]) / 2
    cby = (boxB[1] + boxB[3]) / 2
    return ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5


def greedy_match(tracks, detections, iou_thresh: float):
    """
    Greedy IOU matching: tracky vs detekce.
    Vrací (pairs, unmatched_track_idx, unmatched_detection_idx).
    Předpokládá, že tracky mají metodu `get_box()` vracející [x1,y1,x2,y2].
    Detekce mohou být tuple/listy -- berou se první 4 prvky jako box.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    iou_mat = np.array(
        [[iou(t.get_box(), d[:4]) for d in detections] for t in tracks]
    )
    matched_t = set()
    matched_d = set()
    pairs = []
    while True:
        max_val = iou_mat.max()
        if max_val < iou_thresh:
            break
        t_idx, d_idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        pairs.append((t_idx, d_idx))
        matched_t.add(t_idx)
        matched_d.add(d_idx)
        iou_mat[t_idx, :] = -1
        iou_mat[:, d_idx] = -1

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    return pairs, unmatched_t, unmatched_d
