from dataclasses import dataclass


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int

    confidence: float
    label: str