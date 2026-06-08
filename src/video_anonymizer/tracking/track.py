"""Track dataclass — datová struktura trackeru (per-track state)."""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Track:
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float = 0.0
    lost_frames: int = 0
    hits: int = 1
    age: int = 0
    active: bool = True
    label: str = ""

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
