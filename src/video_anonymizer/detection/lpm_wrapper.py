"""LPM SDK wrapper — primární detektor v pipeline (per assignment: externals/LPM/).

Vyžaduje, aby v `externals/LPM/wrappers/python/` (nebo na LPM_SDK_PATH) byly
`lpm.py` a `er.py` (cffi bindings dodávané s LPM SDK 7.x), a aby v
`externals/LPM/LPM/lib/x64/` bylo `lpm-v7.dll` + `libopenblas.dll`.
Moduly detektoru musí být v `externals/LPM/modules-v7/x64/`.

Pokud LPM není k dispozici, detektor vrací prázdný seznam (pipeline běží dál,
jen nic nedetekuje).
"""
import os
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detection_model import Detection

log = logging.getLogger("video_anonymizer.lpm_wrapper")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SDK_PATH = PROJECT_ROOT / "externals" / "LPM"


def _resolve_sdk_paths():
    """Najde cestu k LPM SDK: env LPM_SDK_PATH > externals/LPM/ > ./LPM/."""
    env = os.environ.get("LPM_SDK_PATH")
    if env:
        return Path(env)
    candidate = DEFAULT_SDK_PATH
    if candidate.is_dir():
        return candidate
    if (PROJECT_ROOT / "LPM").is_dir():
        return PROJECT_ROOT / "LPM"
    return None


def _import_lpm_bindings(sdk_path):
    """Přidá wrappers/python do sys.path a importne lpm/er."""
    wp = sdk_path / "wrappers" / "python"
    if not wp.is_dir():
        wp = sdk_path
    sys.path.insert(0, str(wp))
    from lpm import LPM, LpmBoundingBox, LpmModuleConfig  # noqa: E402
    from er import ER  # noqa: E402
    return LPM, LpmBoundingBox, LpmModuleConfig, ER


class LPMDetector:
    def __init__(self, config, lpm_lib=None, modules_dir=None, view_config=None):
        self.config = config or {}
        self.module_id = self.config.get("module_id", 802)
        self.lpm_version = self.config.get("version", 7)
        self._module_index = -1
        self._ffi = None
        self._er = None
        self._lpm = None

        sdk_path = _resolve_sdk_paths()
        if sdk_path is None:
            log.warning("LPM SDK not found — LPMDetector will be inert. "
                        "Set LPM_SDK_PATH or place SDK at externals/LPM/")
            return

        try:
            LPM, LpmBoundingBox, LpmModuleConfig, ER = _import_lpm_bindings(sdk_path)
        except Exception as e:
            log.warning("Could not import LPM cffi bindings from %s: %s", sdk_path, e)
            return

        if lpm_lib is None:
            lpm_lib = sdk_path / "LPM" / "lib" / "x64" / f"lpm-v{self.lpm_version}.dll"
        if modules_dir is None:
            modules_dir = sdk_path / f"modules-v{self.lpm_version}" / "x64"
        if view_config is None:
            view_config = (sdk_path / f"modules-v{self.lpm_version}"
                           / "config_camera_view_generic.ini")

        # Podpůrné knihovny (libopenblas.dll, …) ze stejného adresáře jako
        # lpm-v7.dll. Bez nich SDK neprovádí výpočty a vrací prázdné výsledky.
        libs_dir = lpm_lib.parent
        lib_suffixes = (".dll", ".so", ".dylib")
        support_libs = [str(p) for p in libs_dir.iterdir()
                        if p.suffix.lower() in lib_suffixes
                        and "lpm" not in p.name.lower()]
        if support_libs:
            log.info("LPM support libs: %s",
                     ", ".join(Path(p).name for p in support_libs))

        try:
            from cffi import FFI
            ffi = FFI()
            er = ER(ffi, str(lpm_lib), support_libs)
            lpm = LPM(ffi, str(lpm_lib), support_libs)
            lpm.init_lpm(str(modules_dir))
            view = lpm.load_view_config(str(view_config))
            mod_cfg = LpmModuleConfig()
            mod_cfg.ocr_compute_on_gpu = False
            mod_cfg.ocr_num_threads = 1
            mod_cfg.det_compute_on_gpu = False
            mod_cfg.det_num_threads = 1
            midx = lpm.get_module_index(self.module_id, 0, 0)
            if midx < 0:
                log.error("Module %d not found in %s", self.module_id, modules_dir)
                return
            lpm.load_module(midx, view, mod_cfg)
            self._ffi = ffi
            self._er = er
            self._lpm = lpm
            self._LpmBoundingBox = LpmBoundingBox
            self._module_index = midx
            log.info("LPM initialized: module %d, version %d", self.module_id, self.lpm_version)
        except Exception as e:
            log.error("LPM init failed: %s", e)

    def _run_lpm(self, frame_bgr):
        if self._lpm is None:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        er_image = self._er.convert_pil_image_to_erimage(Image.fromarray(rgb))
        bb = self._LpmBoundingBox()
        bb.top_left_col = 0
        bb.top_left_row = 0
        bb.bot_right_col = er_image.width - 1
        bb.bot_right_row = er_image.height - 1
        res = self._lpm.run_detection_module(self._module_index, er_image, bb)
        out = []
        for det in res.detections:
            x1, y1 = int(det.position.top_left_col), int(det.position.top_left_row)
            x2, y2 = int(det.position.bot_right_col), int(det.position.bot_right_row)
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            out.append(Detection(x=x1, y=y1, w=w, h=h,
                                 confidence=float(det.confidence),
                                 label="Face"))
        return out

    def detect(self, frame_bgr):
        """Spustí LPM detekci; vrací list[Detection]."""
        try:
            return self._run_lpm(frame_bgr)
        except Exception as e:
            log.error("LPM detect failed: %s", e)
            return []

    def close(self):
        if self._lpm is not None and hasattr(self._lpm, "close"):
            try:
                self._lpm.close()
            except Exception:
                pass
