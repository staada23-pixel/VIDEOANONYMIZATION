"""Per-frame recording of detection/tracking state for forward-pass logging and backward fill."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict


@dataclass
class TrackFrame:
    id: int
    box: list[float]
    conf: float
    state: str
    source: str
    landmarks: list[list[float]] | None = None


@dataclass
class FrameSnapshot:
    frame_idx: int
    tracks: list[TrackFrame]
    cam_dx: float
    cam_dy: float
    has_raw_detection: bool


class FrameHistory:
    def __init__(self):
        self.snapshots: list[FrameSnapshot] = []

    def record(self, frame_idx, tracks, cam_dx, cam_dy, raw_detection_count):
        has_raw = raw_detection_count > 0
        track_frames = []
        for t in tracks:
            state_val = t.state.value
            src = "detection" if state_val in ("active", "low") else "kcf"
            lm = None
            if hasattr(t, "landmarks") and t.landmarks:
                lm = [[float(v) for v in p] for p in t.landmarks]
            track_frames.append(TrackFrame(
                id=t.id,
                box=[float(v) for v in t.box],
                conf=float(t.conf),
                state=state_val,
                source=src,
                landmarks=lm,
            ))
        self.snapshots.append(FrameSnapshot(
            frame_idx=frame_idx,
            tracks=track_frames,
            cam_dx=float(cam_dx),
            cam_dy=float(cam_dy),
            has_raw_detection=has_raw,
        ))

    def to_dict(self) -> dict:
        return {
            "n_frames": len(self.snapshots),
            "frames": [asdict(s) for s in self.snapshots],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> FrameHistory:
        hist = cls()
        for fd in data["frames"]:
            tracks = [TrackFrame(**t) for t in fd["tracks"]]
            hist.snapshots.append(FrameSnapshot(
                frame_idx=fd["frame_idx"],
                tracks=tracks,
                cam_dx=fd["cam_dx"],
                cam_dy=fd["cam_dy"],
                has_raw_detection=fd["has_raw_detection"],
            ))
        return hist
