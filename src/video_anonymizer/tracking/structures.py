"""Datové struktury pro tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base_tracker import BaseTracker


class TrackState(Enum):
    """Stav tracku v ByteTrackeru."""
    ACTIVE = "active"   # spárováno se silnou detekcí
    LOW = "low"         # spárováno se slabou detekcí
    LOST = "lost"       # bez detekce — KCF/CMC drží


@dataclass
class Track:
    """Jeden tracked objekt s přiřazeným trackerem (KCF)."""
    id: int
    box: list                       # [x1, y1, x2, y2]
    conf: float
    state: TrackState
    lost_frames: int = 0
    landmarks: list | None = None   # 5 YuNet landmarků
    confidence_score: float = 0.3   # 0–1, roste s detekcema, klesá při ztrátě
    kcf_ok: bool = True
    kcf_psr: float = 99.0
    kcf_template_score: float = 0.0
    tracker: Optional["BaseTracker"] = field(default=None, repr=False)
