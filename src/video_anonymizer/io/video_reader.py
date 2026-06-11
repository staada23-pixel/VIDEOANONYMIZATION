"""Video reader - obal nad cv2.VideoCapture s iterátorem pres snimky."""
from __future__ import annotations

import os

import cv2


class VideoReader:
    """
    Cte video ze souboru nebo webcam zarizeni.
    `source` muze byt:
      - str (cesta k .mp4, .avi, ...)
      - int (index kamery, typicky 0)
    """

    def __init__(self, source):
        # Pokud je to string ale vypada jako int (vc. zapornych), preved na int
        if isinstance(source, str):
            s = source.strip()
            if s.lstrip("-").isdigit():
                source = int(s)

        # Pro string cesty: fallback hledani v par smysluplnych umistenich
        if isinstance(source, str) and not os.path.exists(source):
            here = os.path.dirname(os.path.abspath(__file__))
            for candidate in (
                os.path.join(os.getcwd(), source),
                os.path.join(here, "..", "..", source),                  # project/
                os.path.join(here, "..", "..", "..", source),            # EYEDEA PROJECT/
            ):
                if os.path.isfile(candidate):
                    source = candidate
                    break

        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Nelze otevrit video: {source!r}. "
                f"Zkontroluj cestu nebo zda soubor existuje."
            )

    # Metadata

    @property
    def fps(self) -> float:
        v = float(self._cap.get(cv2.CAP_PROP_FPS))
        return v if v > 0 else 25.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)

    def __len__(self) -> int:
        if self.is_webcam:
            return 0  # webcamy nemaji smysluplny frame_count
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(0, n)

    def __iter__(self):
        self._frame_idx = 0
        return self

    def __next__(self) -> tuple[int, "np.ndarray"]:
        ok, frame = self._cap.read()
        if not ok:
            raise StopIteration
        self._frame_idx += 1
        return self._frame_idx, frame

    def get_frame(self, frame_idx: int):
        """Vrátí BGR frame na daném indexu (seek). None pokud neexistuje."""
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
