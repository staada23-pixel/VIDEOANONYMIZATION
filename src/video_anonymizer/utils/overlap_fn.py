def iou(b1, b2):
    """Intersection over Union dvou bounding boxů."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def apply_nms(detections_with_conf, iou_threshold=0.3):
    """Odstraní duplicitní boxy — včetně těch které jsou uvnitř jiného boxu."""
    if not detections_with_conf:
        return []
    detections_with_conf = sorted(detections_with_conf, key=lambda x: x[1], reverse=True)
    kept = []
    while detections_with_conf:
        best = detections_with_conf.pop(0)
        kept.append(best)
        bx1, by1, bx2, by2 = best[0]
        remaining = []
        for d in detections_with_conf:
            dx1, dy1, dx2, dy2 = d[0]
            if iou(best[0], d[0]) >= iou_threshold:
                continue
            inter_area = max(0, min(bx2, dx2) - max(bx1, dx1)) * max(0, min(by2, dy2) - max(by1, dy1))
            d_area = (dx2 - dx1) * (dy2 - dy1)
            if d_area > 0 and inter_area / d_area > 0.7:
                continue
            remaining.append(d)
        detections_with_conf = remaining
    return kept


def is_likely_face(bbox, frame_width, frame_height, min_face_ratio=0.03):
    """Filtruje non-face objekty podle poměru stran a velikosti."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if h <= 0 or w <= 0:
        return False
    aspect_ratio = w / h
    if aspect_ratio < 0.4 or aspect_ratio > 2.0:
        return False
    min_size = min(frame_width, frame_height) * min_face_ratio
    if w < min_size or h < min_size:
        return False
    if w > frame_width * 0.6 or h > frame_height * 0.6:
        return False
    return True