"""Wrapper kolem Eyedea LPM SDK — inicializace, detekce, cleanup.

Importy `er` a `lpm` jsou lazy (v __init__), aby se daly pouzivat i fallback
detektory (face/hog) bez pritomnosti Eyedea SDK na PATH.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from cffi import FFI
from PIL import Image

from .structures import Detection

# Tyto se naplni v __init__ (lazy import)
ER = None
LPM = None
LpmBoundingBox = None
LpmModuleConfig = None


def _ensure_lpm_imports():
    """Pokus se importovat Eyedea SDK wrappery z projektu / externals."""
    global ER, LPM, LpmBoundingBox, LpmModuleConfig
    if ER is not None and LPM is not None:
        return

    # Hledej er.py a lpm.py v projektu a v externals
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "..", "..")),  # project/
        os.path.abspath(os.path.join(here, "..", "..", "..", "externals")),  # externals/
        os.path.abspath(os.path.join(here, "..", "..", "..", "..")),  # EYEDEA PROJECT/
    ]
    for c in candidates:
        if c not in sys.path and os.path.isdir(c):
            sys.path.insert(0, c)

    try:
        from er import ER as _ER
        from lpm import LPM as _LPM, LpmBoundingBox as _LpmBoundingBox, LpmModuleConfig as _LpmModuleConfig
        ER = _ER
        LPM = _LPM
        LpmBoundingBox = _LpmBoundingBox
        LpmModuleConfig = _LpmModuleConfig
    except ImportError as e:
        raise ImportError(
            "Eyedea LPM SDK není dostupný (chybí er.py/lpm.py). "
            "Přidej cestu k SDK wrapperům do PYTHONPATH nebo do externals/, "
            "nebo použij --detector face/hog/haar (bez HASP).\n"
            f"Původní chyba: {e}"
        )


class LPMWrapper:
    """
    Zapouzdřuje veškerý LPM SDK setup kód:
      - hledání SDK rootu (z configu nebo autodetekce)
      - načtení DLL a podpůrných knihoven
      - init LPM, načtení view configu, načtení modulu
      - běh detekce na snímku
    """

    def __init__(self, config: dict):
        _ensure_lpm_imports()  # lazy import er.py a lpm.py

        lpm_cfg = config.get("lpm", {}) or {}
        sdk_root_override = lpm_cfg.get("sdk_root")
        self.module_index: int = int(lpm_cfg.get("module_index", 0))
        self.compute_on_gpu: bool = bool(lpm_cfg.get("compute_on_gpu", False))
        self.num_threads: int = int(lpm_cfg.get("num_threads", 1))

        sdk_root = self._resolve_sdk_root(sdk_root_override)
        lpm_lib, support_libs, modules_dir, view_config = self._discover_sdk(sdk_root)

        self.ffi = FFI()
        self.eyedea_er = ER(self.ffi, lpm_lib, support_libs)
        self.lpm = LPM(self.ffi, lpm_lib, support_libs)
        self.lpm.init_lpm(modules_dir)
        print("LPM initialized — version {}".format(self.lpm.get_version()))

        self.view_config = self.lpm.load_view_config(view_config)
        self.module_config = LpmModuleConfig()
        self.module_config.ocr_compute_on_gpu = False
        self.module_config.ocr_num_threads = self.num_threads
        self.module_config.det_compute_on_gpu = self.compute_on_gpu
        self.module_config.det_num_threads = self.num_threads

        module_info = self.lpm.get_module_info(self.module_index)
        print(f"Using module: {module_info.name}")
        self.lpm.load_module(self.module_index, self.view_config, self.module_config)

    @staticmethod
    def _resolve_sdk_root(override) -> str | None:
        """Pokud je sdk_root v configu, použij ho; jinak vrať None (autodetekce)."""
        if override:
            return os.path.abspath(override)
        return None

    @staticmethod
    def _discover_sdk(sdk_root: str | None) -> tuple[str, list[str], str, str]:
        """
        Najdi cesty k LPM DLL, support libs, modules dir a view config ini.
        Pokud sdk_root je None, autodetekce:
          Hledá adresář, kde existuje jak `LPM/lib/x64/` (DLL), tak `modules-v7/` (moduly).
          To odpovídá layoutu `LPM-v7.x-.../{LPM,modules-v7}/`.

        Vrací (lpm_lib_path, support_libs_paths, modules_dir, view_config_path).
        """
        lib_prefix = "lib" if os.name == "posix" else ""
        lib_suffix = "dll" if os.name == "nt" else "so"
        arch = "x64" if os.name == "nt" else "x86_64"
        lpm_version = 7

        if sdk_root is None:
            wrapper_dir = os.path.dirname(os.path.abspath(__file__))
            project_parent = os.path.abspath(
                os.path.join(wrapper_dir, "..", "..", "..", "..")
            )
            candidates: list[str] = [
                # wrapper-relative (původní layout)
                os.path.abspath(os.path.join(wrapper_dir, "..", "..", "..", "..", "LPM")),
                os.path.abspath(os.path.join(wrapper_dir, "..", "..", "..", "LPM")),
                os.path.abspath(os.path.join(wrapper_dir, "..", "..", "LPM")),
                # externals v projektu
                os.path.abspath(os.path.join(wrapper_dir, "..", "..", "..", "externals", "LPM")),
            ]
            # Hledej LPM-* v nadřazeném adresáři projektu
            if os.path.isdir(project_parent):
                for entry in os.listdir(project_parent):
                    full = os.path.join(project_parent, entry)
                    if os.path.isdir(full) and entry.lower().startswith("lpm-"):
                        candidates.append(full)

            sdk_root = None
            for c in candidates:
                # Hledáme adresář, kde je buď DLL, nebo moduly
                if os.path.isdir(os.path.join(c, "LPM", "lib", arch)) or \
                   os.path.isdir(os.path.join(c, "lib", arch)) or \
                   os.path.isdir(os.path.join(c, f"modules-v{lpm_version}")):
                    sdk_root = c
                    break

            if sdk_root is None:
                raise FileNotFoundError(
                    "LPM SDK nenalezen. Nastav 'lpm.sdk_root' v config.yaml.\n"
                    "Zkoušel jsem:\n  " + "\n  ".join(candidates)
                )

        sdk_root = os.path.abspath(sdk_root)

        # LPM DLL může být v <sdk_root>/lib/x64/ NEBO <sdk_root>/LPM/lib/x64/
        dll_search_dirs = [
            os.path.join(sdk_root, "lib", arch),
            os.path.join(sdk_root, "LPM", "lib", arch),
        ]
        lpm_lib = None
        libs_dir = None
        for d in dll_search_dirs:
            candidate = os.path.join(d, "{0}lpm-v{1}.{2}".format(lib_prefix, lpm_version, lib_suffix))
            if os.path.isfile(candidate):
                lpm_lib = candidate
                libs_dir = d
                break
        if lpm_lib is None:
            raise FileNotFoundError(
                f"lpm-v{lpm_version}.{lib_suffix} nenalezen v: {dll_search_dirs}"
            )
        support_libs = [
            os.path.join(libs_dir, f) for f in os.listdir(libs_dir)
            if (f.endswith(".dll") or f.endswith(".so")) and "lpm" not in f
        ]

        # Modules dir — zkus oba kořeny (sdk_root i sdk_root/LPM/)
        module_search_roots = [sdk_root, os.path.join(sdk_root, "LPM")]
        possible_modules_dirs: list[str] = []
        for r in module_search_roots:
            possible_modules_dirs += [
                os.path.join(r, f"modules-v{lpm_version}", arch),
                os.path.join(r, f"modules-v{lpm_version}", f"modules-v{lpm_version}", arch),
                os.path.join(r, f"modules-v{lpm_version}"),
                os.path.join(r, f"modules-v{lpm_version}", f"modules-v{lpm_version}"),
            ]
        modules_dir = None
        for candidate in possible_modules_dirs:
            if os.path.isdir(candidate):
                modules_dir = candidate
                break
        if modules_dir is None:
            raise FileNotFoundError(
                f"Modules directory not found. Tried: {possible_modules_dirs}"
            )

        if os.path.basename(modules_dir).lower() == arch.lower():
            view_config = os.path.join(os.path.dirname(modules_dir), "config_camera_view_generic.ini")
        else:
            view_config = os.path.join(modules_dir, "config_camera_view_generic.ini")

        return lpm_lib, support_libs, modules_dir, view_config

    def detect(self, frame_bgr) -> list[Detection]:
        """Spusť detekci na BGR snímku. Vrať list Detection objektů."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        er_image = self.eyedea_er.convert_pil_image_to_erimage(pil_img)

        bbox = LpmBoundingBox()
        bbox.top_left_col = 0
        bbox.top_left_row = 0
        bbox.bot_right_col = er_image.width - 1
        bbox.bot_right_row = er_image.height - 1
        results = self.lpm.run_detection_module(self.module_index, er_image, bbox)

        detections: list[Detection] = []
        for det in results.detections:
            x1 = float(det.position.top_left_col)
            y1 = float(det.position.top_left_row)
            x2 = float(det.position.bot_right_col)
            y2 = float(det.position.bot_right_row)
            detections.append(Detection(x1, y1, x2, y2, float(det.confidence)))
        return detections

    def close(self) -> None:
        """Uvolni LPM SDK."""
        self.lpm.close()
