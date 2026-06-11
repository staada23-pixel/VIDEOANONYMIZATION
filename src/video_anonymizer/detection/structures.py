"""Datové struktury pro detekci."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """Jedna LPM detekce — obdélník + confidence + landmarky."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)

    def to_tuple(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2, self.confidence)

    @property
    def box(self) -> list:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def conf(self) -> float:
        """Zkrácený alias pro confidence."""
        return self.confidence
