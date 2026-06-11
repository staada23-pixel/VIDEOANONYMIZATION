"""KCF tracker s Gaussian kernelem (vlastní implementace) + template-matching fallback."""
from __future__ import annotations

import cv2
import numpy as np

from .base_tracker import BaseTracker


# ── Pomocné matematické funkce (čisté, side-effect free) ────────────

def _hann_window(sz) -> np.ndarray:
    """2D Hann okno pro potlačení boundary efektů."""
    h, w = sz
    wh = np.hanning(h).reshape(-1, 1)
    ww = np.hanning(w).reshape(1, -1)
    return wh * ww


def _gaussian_labels(sz, sigma: float) -> np.ndarray:
    """Cílová Gaussian mapa — peak uprostřed."""
    h, w = sz
    cy, cx = h // 2, w // 2
    ys = np.arange(h) - cy
    xs = np.arange(w) - cx
    xx, yy = np.meshgrid(xs, ys)
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


def _extract_features(patch: np.ndarray, hann: np.ndarray) -> np.ndarray:
    """Předzpracování patche: grayscale + normalizace + Hann okno."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray * hann


def _gaussian_kernel_correlation(xf, yf, sigma: float) -> np.ndarray:
    """Gaussian kernel v Fourierově prostoru (Henriques et al. 2015)."""
    n = xf.size
    xx = np.real(np.sum(xf * np.conj(xf))) / n
    yy = np.real(np.sum(yf * np.conj(yf))) / n
    xyf = xf * np.conj(yf)
    xy = np.real(np.fft.ifft2(xyf))
    kf = np.fft.fft2(
        np.exp(-1.0 / (sigma ** 2) * np.maximum(0, (xx + yy - 2 * xy) / n))
    )
    return kf


# ── Hlavní třída ───────────────────────────────────────────────────

class KCFTracker(BaseTracker):
    """Kernel Correlation Filter tracker s Gaussian kernelem a template fallbackem."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        # KCF parametry
        self.padding: float = float(cfg.get("padding", 2.0))
        self.sigma: float = float(cfg.get("sigma", 0.5))
        self.lambda_reg: float = float(cfg.get("lambda", 1e-4))
        self.learning_rate: float = float(cfg.get("learning_rate", 0.075))
        self.output_sigma: float = float(cfg.get("output_sigma", 0.1))
        self.interp_factor: float = float(cfg.get("interp_factor", 0.075))

        # Drift detekce
        self.psr_threshold: float = float(cfg.get("psr_threshold", 7.0))
        self.max_speed: float = float(cfg.get("max_speed", 0.35))

        # Template-matching fallback
        self.tm_search_mult: float = float(cfg.get("tm_search_mult", 3.0))
        self.tm_min_score: float = float(cfg.get("tm_min_score", 0.35))
        self.tm_refresh_kcf: float = float(cfg.get("tm_refresh_kcf", 0.15))
        self.tm_refresh_fallback: float = float(cfg.get("tm_refresh_fallback", 0.30))
        self.tm_max_jump: float = float(cfg.get("tm_max_jump", 2.0))

        # Obecné
        self.min_box_size: int = int(cfg.get("min_box_size", 10))

        # Vnitřní stav
        self.box: list | None = None
        self._alive: bool = False
        self._ok: bool = True
        self._xf = None
        self._alphaf = None
        self._sz = None
        self._hann = None
        self._yf = None
        self._psr: float = 99.0

        # Template fallback
        self._tmpl: np.ndarray | None = None
        self._tmpl_sz: tuple | None = None
        self._tmpl_score: float = 0.0

        # Geometrie boxu
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._bw: int = 0
        self._bh: int = 0

    # ── Vlastnosti z BaseTracker ─────────────────────────

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def is_ok(self) -> bool:
        return self._ok

    @property
    def psr(self) -> float:
        return self._psr

    @property
    def template_score(self) -> float:
        return self._tmpl_score

    # ── Public API ──────────────────────────────────────

    def init(self, frame_bgr, box) -> None:
        """Inicializuj KCF na daném boxu."""
        x1, y1, x2, y2 = [int(v) for v in box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if bw < self.min_box_size or bh < self.min_box_size:
            self._alive = False
            return

        # Search okno = padding * velikost boxu, zarovnáno na sudé
        sw = int(bw * self.padding) // 2 * 2
        sh = int(bh * self.padding) // 2 * 2
        self._sz = (sh, sw)
        self._hann = _hann_window(self._sz).astype(np.float32)

        # Cílová Gaussian
        sigma_px = self.output_sigma * np.sqrt(sw * sh)
        labels = _gaussian_labels(self._sz, sigma_px)
        self._yf = np.fft.fft2(labels).astype(np.complex64)

        # Střed boxu
        self._cx = x1 + bw / 2
        self._cy = y1 + bh / 2
        self._bw = bw
        self._bh = bh

        patch = self._get_patch(frame_bgr)
        if patch is None:
            self._alive = False
            return

        xf = np.fft.fft2(_extract_features(patch, self._hann))
        kf = _gaussian_kernel_correlation(xf, xf, self.sigma)
        self._alphaf = self._yf / (kf + self.lambda_reg)
        self._xf = xf

        # Ulož i grayscale template pro fallback tracking
        self._tmpl = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self._tmpl_sz = patch.shape[:2]
        self._tmpl_score = 1.0

        self.box = [x1, y1, x2, y2]
        self._alive = True
        self._ok = True

    def refresh_template(self, frame_bgr) -> None:
        """Vynucený refresh template z aktuální pozice (po LPM confirm)."""
        patch = self._get_patch(frame_bgr)
        if patch is not None:
            self._tmpl = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            self._tmpl_sz = patch.shape[:2]

    def apply_camera_motion(self, dx: float, dy: float) -> None:
        """Posuň tracker (střed + box) o kompenzovaný pohyb kamery."""
        if self._alive:
            self._cx += dx
            self._cy += dy
            if self.box is not None:
                self.box = [
                    self.box[0] + dx, self.box[1] + dy,
                    self.box[2] + dx, self.box[3] + dy,
                ]

    def get_box(self) -> list:
        return self.box if self.box is not None else [0, 0, 0, 0]

    # ── Internals ───────────────────────────────────────

    def _get_patch(self, frame_bgr):
        """Vyřízne search okno kolem středu a přeškáluje na _sz."""
        h, w = frame_bgr.shape[:2]
        sh, sw = self._sz
        x1 = int(self._cx - sw / 2)
        y1 = int(self._cy - sh / 2)
        x2 = x1 + sw
        y2 = y1 + sh

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w, x2)
        y2c = min(h, y2)

        if x2c <= x1c or y2c <= y1c:
            return None

        patch = frame_bgr[y1c:y2c, x1c:x2c]
        if pad_l or pad_t or pad_r or pad_b:
            patch = cv2.copyMakeBorder(
                patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )
        if patch.shape[0] != sh or patch.shape[1] != sw:
            patch = cv2.resize(patch, (sw, sh))
        return patch

    def _compute_psr(self, response: np.ndarray) -> float:
        """PSR = (peak - mean_sidelobe) / std_sidelobe."""
        peak_val = response.max()
        py, px = np.unravel_index(response.argmax(), response.shape)
        r = 5
        mask = np.ones_like(response, dtype=bool)
        mask[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = False
        sidelobe = response[mask]
        if sidelobe.size == 0:
            return 0.0
        return float((peak_val - sidelobe.mean()) / (sidelobe.std() + 1e-6))

    def _track_template(self, frame_bgr) -> bool:
        """Template matching se širokým search oknem — fallback když KCF driftuje."""
        if self._tmpl is None or self._tmpl_sz is None:
            return False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        th, tw = self._tmpl_sz
        bw, bh = self._bw, self._bh

        sw = int(bw * self.tm_search_mult)
        sh = int(bh * self.tm_search_mult)
        x1 = int(self._cx - sw / 2)
        y1 = int(self._cy - sh / 2)
        x2 = x1 + sw
        y2 = y1 + sh

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w, x2)
        y2c = min(h, y2)

        if x2c - x1c < tw or y2c - y1c < th:
            return False

        search = gray[y1c:y2c, x1c:x2c]
        if pad_l or pad_t or pad_r or pad_b:
            search = cv2.copyMakeBorder(
                search, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
            )
        if search.shape[0] < th or search.shape[1] < tw:
            return False

        result = cv2.matchTemplate(search, self._tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        self._tmpl_score = float(max_val)

        if max_val < self.tm_min_score:
            return False

        new_cx = x1 + max_loc[0] + tw / 2
        new_cy = y1 + max_loc[1] + th / 2

        move_x = abs(new_cx - self._cx)
        move_y = abs(new_cy - self._cy)
        max_move = max(bw, bh) * self.tm_max_jump
        if move_x > max_move or move_y > max_move:
            return False

        self._cx = new_cx
        self._cy = new_cy
        self.box = [
            new_cx - bw / 2, new_cy - bh / 2,
            new_cx + bw / 2, new_cy + bh / 2,
        ]

        # Adaptuj template na pomalu se měnící vzhled
        new_patch = self._get_patch(frame_bgr)
        if new_patch is not None:
            new_gray = cv2.cvtColor(new_patch, cv2.COLOR_BGR2GRAY)
            if new_gray.shape == self._tmpl.shape:
                self._tmpl = cv2.addWeighted(
                    self._tmpl, 1.0 - self.tm_refresh_fallback,
                    new_gray, self.tm_refresh_fallback, 0,
                )
        return True

    # ── Hlavní tracking krok ───────────────────────────

    def update(self, frame_bgr) -> bool:
        """
        1. KCF — pokud PSR dobrý a skok rozumný, použij KCF pozici.
        2. Jinak template matching se širokým oknem.
        3. Jinak freeze, čekej na LPM reinit.
        """
        if not self._alive:
            return False

        patch = self._get_patch(frame_bgr)
        if patch is None:
            self._alive = False
            return False

        # 1. KCF tracking
        zf = np.fft.fft2(_extract_features(patch, self._hann))
        kzf = _gaussian_kernel_correlation(zf, self._xf, self.sigma)
        response = np.real(np.fft.ifft2(self._alphaf * kzf))
        self._psr = self._compute_psr(response)

        py, px = np.unravel_index(response.argmax(), response.shape)
        sh, sw = self._sz
        dy = py if py < sh // 2 else py - sh
        dx = px if px < sw // 2 else px - sw
        kcf_cx = self._cx + dx
        kcf_cy = self._cy + dy

        move_x = abs(kcf_cx - self._cx)
        move_y = abs(kcf_cy - self._cy)
        is_jump = (
            move_x > self._bw * self.max_speed
            or move_y > self._bh * self.max_speed
        )
        is_low_psr = self._psr < self.psr_threshold
        is_no_sig = self._psr < 3.0
        kcf_trustworthy = (not is_jump) and (not is_no_sig)

        # 2. Rozhodni o pozici
        if kcf_trustworthy:
            self._cx = kcf_cx
            self._cy = kcf_cy
            self.box = [
                kcf_cx - self._bw / 2, kcf_cy - self._bh / 2,
                kcf_cx + self._bw / 2, kcf_cy + self._bh / 2,
            ]
            # Pomalu adaptuj template
            if self._tmpl is not None:
                new_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                if new_gray.shape == self._tmpl.shape:
                    self._tmpl = cv2.addWeighted(
                        self._tmpl, 1.0 - self.tm_refresh_kcf,
                        new_gray, self.tm_refresh_kcf, 0,
                    )
                    self._tmpl_score = 1.0
        else:
            # KCF nedůvěryhodný — zkus template matching
            tm_ok = self._track_template(frame_bgr)
            if not tm_ok:
                # Ani template nepomohl — freeze
                self._ok = False
                return True

        # 3. Vyhodnoť důvěryhodnost, případně zmraz model
        if is_low_psr or is_jump or (not kcf_trustworthy):
            self._ok = False
            return True

        # Vše ok — obnov _ok a aktualizuj KCF model
        self._ok = True
        xf_new = np.fft.fft2(_extract_features(patch, self._hann))
        kf_new = _gaussian_kernel_correlation(xf_new, xf_new, self.sigma)
        alphaf_new = self._yf / (kf_new + self.lambda_reg)
        self._alphaf = (
            (1 - self.interp_factor) * self._alphaf
            + self.interp_factor * alphaf_new
        )
        self._xf = (
            (1 - self.learning_rate) * self._xf
            + self.learning_rate * xf_new
        )
        return True
