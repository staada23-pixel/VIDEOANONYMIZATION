import cv2


class CSRTTracker:
    """
    Wrapper nad OpenCV CSRT trackerem.
    Sleduje vizuální vzhled objektu — vhodné pro pohyblivou kameru (bodycam).
    """
    _id_counter = 0

    def __init__(self, frame_bgr, bbox_x1y1x2y2, confidence):
        CSRTTracker._id_counter += 1
        self.id = CSRTTracker._id_counter
        self.confidence = confidence
        self.lost_frames = 0
        self.active = True

        self.tracker = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = bbox_x1y1x2y2
        self.tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        self.bbox = bbox_x1y1x2y2

    def update(self, frame_bgr):
        ok, rect = self.tracker.update(frame_bgr)
        if ok:
            x, y, w, h = [int(v) for v in rect]
            self.bbox = (x, y, x + w, y + h)
            self.active = True
            self.lost_frames = 0
        else:
            self.active = False
            self.lost_frames += 1
        return ok

    def reinit(self, frame_bgr, bbox_x1y1x2y2, confidence):
        x1, y1, x2, y2 = bbox_x1y1x2y2
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        self.bbox = bbox_x1y1x2y2
        self.confidence = confidence
        self.active = True
        self.lost_frames = 0

    def get_bbox(self):
        return self.bbox

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0