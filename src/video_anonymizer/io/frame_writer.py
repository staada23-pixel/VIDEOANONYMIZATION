"""Frame writer — ukládá BGR snímky do adresáře."""
from __future__ import annotations

import os

import cv2


class FrameWriter:
    """
    Ukládá snímky jako `frame_XXXXX.jpg` do output_dir.
    Pokud `save_only_with_detections`, ukládá jen snímky kde `has_detections=True`.
    `save_every_n` řídí subsampling (1 = vše, 2 = každý druhý, ...).
    """

    def __init__(
        self,
        output_dir: str,
        save_only_with_detections: bool = True,
        save_every_n: int = 1,
    ):
        self.output_dir = output_dir
        self.save_only_with_detections = save_only_with_detections
        self.save_every_n = max(1, int(save_every_n))
        os.makedirs(self.output_dir, exist_ok=True)
        self._saved = 0

    def write(self, frame_idx: int, frame_bgr, has_detections: bool) -> bool:
        """Ulož snímek pokud splňuje podmínky. Vrať True pokud byl uložen."""
        if frame_idx % self.save_every_n != 0:
            return False
        if self.save_only_with_detections and not has_detections:
            return False
        out_path = os.path.join(self.output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, frame_bgr)
        self._saved += 1
        return True

    @property
    def saved_count(self) -> int:
        return self._saved
