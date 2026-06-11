"""Abstraktní base class pro trackery."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """
    Rozhraní trackeru, který ByteTracker používá.
    Konkrétní implementace (KCF, CSRT, MIL, …) dědí odtud.
    """

    def refresh_template(self, frame_bgr) -> None:
        """Volitelne: obnov referencni template. Bezna implementace = no-op."""

    @abstractmethod
    def init(self, frame_bgr, box: list) -> None:
        """Inicializuj tracker na daném boxu v daném snímku."""

    @abstractmethod
    def update(self, frame_bgr) -> bool:
        """Jeden tracking krok. Vrať False pokud tracker definitivně selhal."""

    @abstractmethod
    def get_box(self) -> list:
        """Vrať aktuální box [x1, y1, x2, y2]."""

    @abstractmethod
    def apply_camera_motion(self, dx: float, dy: float) -> None:
        """Posuň tracker o kompenzovaný pohyb kamery."""

    @property
    @abstractmethod
    def alive(self) -> bool:
        """Je tracker ještě naživu?"""

    @property
    @abstractmethod
    def is_ok(self) -> bool:
        """False = drift detekován, čeká na reinit."""

    @property
    @abstractmethod
    def psr(self) -> float:
        """Aktuální PSR (nebo obdobná jistota)."""

    @property
    @abstractmethod
    def template_score(self) -> float:
        """Skóre template fallbacku (0..1, 0 = nezainicializováno)."""
