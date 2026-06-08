"""Frame writing — ukládání výstupů v různých formátech.

Tři writer typy:
  - FrameWriter:   sekvence JPEG/PNG na disk (frame_NNNNNN.jpg)
  - VideoWriter:   mp4/avi přes cv2.VideoWriter
  - JSONWriter:    per-frame metadata (detekce, track bbox, anonymizované bboxy)

`MultiWriter` umožňuje kombinovat více writerů najednou.
"""
from pathlib import Path
import json
import cv2


def save_frame(out_dir, frame_idx, frame, ext="jpg", quality=90):
    """Uloží frame jako frame_<frame_idx:06d>.<ext>. Vrací cestu."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"frame_{frame_idx:06d}.{ext}"
    path = out_dir / name
    if ext.lower() in ("jpg", "jpeg"):
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(path), frame)
    return path


class FrameWriter:
    """Průběžný zápis snímků do adresáře."""

    def __init__(self, out_dir, ext="jpg", quality=90):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ext = ext
        self.quality = quality

    def write(self, frame_idx, frame):
        return save_frame(self.out_dir, frame_idx, frame,
                          ext=self.ext, quality=self.quality)

    def close(self):
        pass


class VideoWriter:
    """Průběžný zápis snímků do video souboru (.mp4/.avi)."""

    def __init__(self, out_path, fps, frame_size, codec="mp4v"):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.suffix:
            out_path = out_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {out_path}")
        self.out_path = out_path
        self.frames_written = 0

    def write(self, frame_idx, frame):
        self._writer.write(frame)
        self.frames_written += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class JSONWriter:
    """Per-frame metadata: detekce, tracker, anonymizované bboxy.

    Formát: {"video": "...", "frames": [{"frame": 0, "detections": [...], "track": {...}}, ...]}
    """

    def __init__(self, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self._records = []
        self._meta = {}

    def set_meta(self, **kwargs):
        self._meta.update(kwargs)

    def write(self, frame_idx, frame=None, detections=None, track=None,
              anonymized_bbox=None):
        record = {"frame": frame_idx}
        if detections is not None:
            record["detections"] = [
                {
                    "x": d.x, "y": d.y, "w": d.w, "h": d.h,
                    "confidence": d.confidence, "label": d.label,
                } for d in detections
            ]
        if track is not None:
            record["track"] = {
                "id": track.id,
                "bbox": list(track.bbox),
                "confidence": track.confidence,
                "lost_frames": track.lost_frames,
                "active": track.active,
            }
        if anonymized_bbox is not None:
            record["anonymized_bbox"] = list(anonymized_bbox)
        self._records.append(record)

    def close(self):
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": self._meta, "frames": self._records},
                      f, ensure_ascii=False, indent=2)


class MultiWriter:
    """Zapisuje do více writerů najednou."""

    def __init__(self, *writers):
        self.writers = [w for w in writers if w is not None]

    def write(self, frame_idx, frame, **kwargs):
        for w in self.writers:
            if isinstance(w, (FrameWriter, VideoWriter)):
                w.write(frame_idx, frame)
            elif isinstance(w, JSONWriter):
                w.write(frame_idx, **kwargs)

    def close(self):
        for w in self.writers:
            w.close()


def make_writer(spec, fps=None, frame_size=None):
    """Factory podle spec dict: {"frames": "dir/", "video": "out.mp4", "json": "meta.json"}."""
    writers = []
    if spec.get("frames"):
        writers.append(FrameWriter(spec["frames"]))
    if spec.get("video") and fps and frame_size:
        writers.append(VideoWriter(spec["video"], fps, frame_size))
    if spec.get("json"):
        writers.append(JSONWriter(spec["json"]))
    if not writers:
        return None
    if len(writers) == 1:
        return writers[0]
    return MultiWriter(*writers)
