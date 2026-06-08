"""video_anonymizer.py — single-file CLI pro video anonymization.

Postaveno na fungujícím video_detection.py z LPM SDK, rozšířeno o:
  * volitelný tracker (CSRT / KCF / Kalman)
  * volitelnou metodu anonymizace (pixelate / gaussian / blackout / none)
  * volitelný vstup, výstup, min-confidence, expanzi bboxu
  * volitelné vykreslení bboxu (pro ladění)

Použití:
    python video_anonymizer.py video.mp4
    python video_anonymizer.py video.mp4 --tracker kcf --blur-method gaussian
    python video_anonymizer.py --help

LPM SDK se hledá v pořadí:
    1. --lpm-sdk PATH
    2. $LPM_SDK_PATH
    3. ../LPM (relativně k tomuto souboru)
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from cffi import FFI
from PIL import Image


# ─── LOGGING ──────────────────────────────────────────────────────
log = logging.getLogger("video_anonymizer")


# ─── CLI DEFAULTS ─────────────────────────────────────────────────
DEFAULT_LPM_VERSION = 7
DEFAULT_MODULE_ID = 802
DEFAULT_MIN_CONFIDENCE = 0.40
DEFAULT_REDETECT_EVERY = 10
DEFAULT_LOST_FRAMES = 15
DEFAULT_SEARCH_RADIUS = 200
DEFAULT_BBOX_EXPAND = 0.15
DEFAULT_PIXEL_BLOCK = 18
DEFAULT_GAUSSIAN_KSIZE = 21
DEFAULT_TRACKER = "csrt"
DEFAULT_BLUR = "pixelate"
DEFAULT_OUTPUT_DIR = "output_video_frames"


# ─── SDK DISCOVERY ────────────────────────────────────────────────
def find_lpm_sdk(arg_path=None):
    """Vrátí (sdk_path, lpm_lib, modules_dir, view_config, wrappers_dir)."""
    candidates = []
    if arg_path:
        candidates.append(Path(arg_path))
    env = os.environ.get("LPM_SDK_PATH")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    # běžné layouty: externals/LPM/, ../LPM/, ./LPM/
    candidates.append(here / "externals" / "LPM")
    candidates.append(here.parent / "LPM")
    candidates.append(here / "LPM")

    for c in candidates:
        if not c or not c.is_dir():
            continue
        # Hledáme LPM/lib/x64/lpm-v7.dll
        lib_prefix = "lib" if os.name == "posix" else ""
        lib_suffix = "dll" if os.name == "nt" else "so"
        arch = "x64" if os.name == "nt" else "x86_64"
        lpm_lib = c / "LPM" / "lib" / arch / f"{lib_prefix}lpm-v{DEFAULT_LPM_VERSION}.{lib_suffix}"
        if not lpm_lib.is_file():
            continue
        # Hledáme modules dir
        mods_root = c / f"modules-v{DEFAULT_LPM_VERSION}"
        if (mods_root / arch).is_dir():
            modules_dir = mods_root / arch
            view_config = mods_root / "config_camera_view_generic.ini"
        elif mods_root.is_dir():
            modules_dir = mods_root
            view_config = mods_root / "config_camera_view_generic.ini"
        else:
            continue
        if not view_config.is_file():
            continue
        wrappers_dir = c / "wrappers" / "python"
        if not (wrappers_dir / "lpm.py").is_file():
            # Fallback: wrappers v SDK rootu
            wrappers_dir = c
        return c, lpm_lib, modules_dir, view_config, wrappers_dir

    raise FileNotFoundError(
        "LPM SDK nenalezen. Nastav LPM_SDK_PATH nebo použij --lpm-sdk."
    )


# ─── HELPERS ──────────────────────────────────────────────────────
def expand_bbox(bbox, factor, frame_shape):
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * factor)),
            max(0, int(y1 - bh * factor)),
            min(w, int(x2 + bw * factor)),
            min(h, int(y2 + bh * factor)))


def center_of(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def clamp_box(box, w, h):
    x, y, ww, hh = box
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    ww = max(1, min(w - x, ww))
    hh = max(1, min(h - y, hh))
    return x, y, ww, hh


# ─── BLUR METHODS ─────────────────────────────────────────────────
def blur_pixelate(frame, bbox, block=18):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh),
                                     interpolation=cv2.INTER_NEAREST)


def blur_gaussian(frame, bbox, ksize=21):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return
    k = ksize | 1  # musí být lichý
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)


def blur_blackout(frame, bbox, color=(0, 0, 0)):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return
    frame[y1:y2, x1:x2] = color


def anonymize_region(frame, bbox, method="pixelate", **kwargs):
    if method == "none":
        return
    if method == "pixelate":
        blur_pixelate(frame, bbox, block=kwargs.get("block", 18))
    elif method == "gaussian":
        blur_gaussian(frame, bbox, ksize=kwargs.get("ksize", 21))
    elif method == "blackout":
        blur_blackout(frame, bbox, color=kwargs.get("color", (0, 0, 0)))
    else:
        raise ValueError(f"Unknown blur method: {method}")


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2, label=None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _lerp_bbox(a, b, t):
    """Lineární interpolace dvou (x1,y1,x2,y2) bboxů; t v [0,1]."""
    if a is None or b is None:
        return b if b is not None else a
    t = max(0.0, min(1.0, t))
    return tuple(int(av + (bv - av) * t) for av, bv in zip(a, b))


def _apply_anonymize(frame, bbox, args, lost_count=0, tracking=True):
    """Aplikuje blur + volitelně bbox na frame (in-place). Vrací frame."""
    if bbox is None:
        return frame
    if tracking and not args.no_blur:
        anon_bbox = expand_bbox(bbox, args.bbox_expand, frame.shape)
        if args.blur_method == "pixelate":
            blur_pixelate(frame, anon_bbox, block=args.pixel_block)
        elif args.blur_method == "gaussian":
            blur_gaussian(frame, anon_bbox, ksize=args.gaussian_ksize)
        elif args.blur_method == "blackout":
            blur_blackout(frame, anon_bbox)
        # "none" → žádná změna
    if args.draw_bbox:
        color = (0, 255, 0) if tracking else (0, 0, 255)
        label = None
        if tracking:
            label = f"id=1 tracker={args.tracker}"
        else:
            label = f"LOST ({lost_count})"
        draw_bbox(frame, bbox, color=color, thickness=2, label=label)
    return frame


# ─── TRACKERS (společné rozhraní) ─────────────────────────────────
class CSRTTracker:
    def __init__(self):
        self._t = None
    def init(self, frame, bbox):
        x, y, w, h = clamp_box(bbox, frame.shape[1], frame.shape[0])
        self._t = cv2.TrackerCSRT_create()
        self._t.init(frame, (x, y, w, h))
        return (x, y, w, h)
    def update(self, frame):
        ok, rect = self._t.update(frame)
        if not ok:
            return False, None
        return True, tuple(int(v) for v in rect)
    def reinit(self, frame, bbox):
        self.init(frame, bbox)


class KCFTracker:
    def __init__(self):
        self._t = None
    def init(self, frame, bbox):
        x, y, w, h = clamp_box(bbox, frame.shape[1], frame.shape[0])
        self._t = cv2.TrackerKCF_create()
        self._t.init(frame, (x, y, w, h))
        return (x, y, w, h)
    def update(self, frame):
        ok, rect = self._t.update(frame)
        if not ok:
            return False, None
        return True, tuple(int(v) for v in rect)
    def reinit(self, frame, bbox):
        self.init(frame, bbox)


class KalmanBoxTracker:
    """8-state: [cx, cy, w, h, dcx, dcy, dw, dh]."""
    def __init__(self, process_noise=0.03, measurement_noise=0.5):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0
        self.kf.transitionMatrix = F
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self._inited = False
    def init(self, frame, bbox):
        x, y, w, h = bbox
        m = np.array([[x + w / 2.0], [y + h / 2.0],
                      [float(w)], [float(h)]], dtype=np.float32)
        self.kf.statePre = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePre[:4] = m
        self.kf.statePost = self.kf.statePre.copy()
        self._inited = True
        return bbox
    def update(self, frame):
        # Kalman je predictor — nikdy neřekne "lost", jen predikuje.
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        w, h = max(1, int(w)), max(1, int(h))
        return True, (int(cx - w / 2), int(cy - h / 2), w, h)
    def correct(self, bbox):
        x, y, w, h = bbox
        m = np.array([[x + w / 2.0], [y + h / 2.0],
                      [float(w)], [float(h)]], dtype=np.float32)
        self.kf.correct(m)
    def reinit(self, frame, bbox):
        self.correct(bbox)


def make_tracker(name):
    name = name.lower()
    if name == "csrt":
        return CSRTTracker()
    if name == "kcf":
        return KCFTracker()
    if name == "kalman":
        return KalmanBoxTracker()
    raise ValueError(f"Unknown tracker: {name}")


# ─── LPM DETECTION ────────────────────────────────────────────────
def run_lpm_detections(lpm, eyedea_er, LpmBoundingBox, module_index,
                       frame_bgr, min_conf):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    er_image = eyedea_er.convert_pil_image_to_erimage(Image.fromarray(rgb))
    bb = LpmBoundingBox()
    bb.top_left_col = 0
    bb.top_left_row = 0
    bb.bot_right_col = er_image.width - 1
    bb.bot_right_row = er_image.height - 1
    res = lpm.run_detection_module(module_index, er_image, bb)
    out = []
    for det in res.detections:
        if det.confidence < min_conf:
            continue
        x1, y1 = int(det.position.top_left_col), int(det.position.top_left_row)
        x2, y2 = int(det.position.bot_right_col), int(det.position.bot_right_row)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append((x1, y1, x2, y2, float(det.confidence)))
    return out


def find_nearest(detections, target_center, max_radius):
    best, best_d2 = None, max_radius * max_radius
    tx, ty = target_center
    for d in detections:
        cx, cy = center_of(d[:4])
        dx, dy = cx - tx, cy - ty
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best, best_d2 = d, d2
    return best


def select_initial_target(frame_bgr, lpm, eyedea_er, LpmBoundingBox,
                          module_index, cap, min_conf,
                          max_scan_frames=30):
    detections = run_lpm_detections(lpm, eyedea_er, LpmBoundingBox,
                                    module_index, frame_bgr, min_conf)
    if detections:
        return detections[0][:4], "lpm-auto", 0

    if cap is None:
        return None, None, 0

    log.info("No face in first frame — scanning next %d frames...",
             max_scan_frames)
    best, best_conf, scanned = None, -1.0, 0
    while scanned < max_scan_frames:
        ok, nxt = cap.read()
        if not ok:
            break
        scanned += 1
        dets = run_lpm_detections(lpm, eyedea_er, LpmBoundingBox,
                                  module_index, nxt, min_conf)
        for d in dets:
            if d[4] > best_conf:
                best, best_conf = d[:4], d[4]
    if best is None:
        return None, None, scanned
    return best, "lpm-auto-scanned", scanned


# ─── ARGUMENT PARSER ──────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="VIDEOANONYMIZATION — CLI (LPM detekce + volitelný tracker + blur)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", nargs="?",
                   help="Cesta ke vstupnímu videu/obrázku. Při vynechání čte ze stdin/interaktivně.")
    p.add_argument("--wizard", action="store_true",
                   help="Spustí interaktivní průvodce (totéž jako spuštění bez argumentů)")
    p.add_argument("--lpm-sdk", help="Cesta ke kořenu LPM SDK (jinak LPM_SDK_PATH nebo default)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Adresář pro výstupní framy (default: {DEFAULT_OUTPUT_DIR})")

    g = p.add_argument_group("Tracker")
    g.add_argument("--tracker", choices=["csrt", "kcf", "kalman"],
                   default=DEFAULT_TRACKER,
                   help=f"Typ trackeru (default: {DEFAULT_TRACKER})")
    g.add_argument("--redetect-every", type=int, default=DEFAULT_REDETECT_EVERY,
                   help=f"Re-detekce každých N framů (default: {DEFAULT_REDETECT_EVERY})")
    g.add_argument("--lost-frames", type=int, default=DEFAULT_LOST_FRAMES,
                   help=f"Po kolika ztracených framech re-akvizice (default: {DEFAULT_LOST_FRAMES})")
    g.add_argument("--search-radius", type=int, default=DEFAULT_SEARCH_RADIUS,
                   help=f"Max vzdálenost detekce od poslední pozice v px (default: {DEFAULT_SEARCH_RADIUS})")

    g = p.add_argument_group("Detector (LPM)")
    g.add_argument("--module-id", type=int, default=DEFAULT_MODULE_ID,
                   help=f"LPM module ID (default: {DEFAULT_MODULE_ID})")
    g.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                   help=f"Minimální confidence LPM (default: {DEFAULT_MIN_CONFIDENCE})")
    g.add_argument("--initial-scan", type=int, default=30,
                   help="Počet framů k proskenování, pokud první frame bez detekce")

    g = p.add_argument_group("Anonymization")
    g.add_argument("--blur-method", choices=["pixelate", "gaussian", "blackout", "none"],
                   default=DEFAULT_BLUR,
                   help=f"Metoda anonymizace (default: {DEFAULT_BLUR})")
    g.add_argument("--pixel-block", type=int, default=DEFAULT_PIXEL_BLOCK,
                   help=f"Velikost bloku pro pixelate (default: {DEFAULT_PIXEL_BLOCK})")
    g.add_argument("--gaussian-ksize", type=int, default=DEFAULT_GAUSSIAN_KSIZE,
                   help=f"Kernel size pro gaussian (default: {DEFAULT_GAUSSIAN_KSIZE}, musí být lichý)")
    g.add_argument("--bbox-expand", type=float, default=DEFAULT_BBOX_EXPAND,
                   help=f"Expanze bboxu před blur (default: {DEFAULT_BBOX_EXPAND})")
    g.add_argument("--no-blur", action="store_true",
                   help="Neprovádět anonymizaci (jen tracker/bbox)")

    g = p.add_argument_group("Output / debug")
    g.add_argument("--draw-bbox", action="store_true",
                   help="Vykreslit zelený bbox kolem sledovaného objektu")
    g.add_argument("--no-preview", action="store_true",
                   help="Bez live preview okna")
    g.add_argument("--save-video", metavar="PATH",
                   help="Uložit výstupní MP4 (kromě framů)")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ─── MAIN ─────────────────────────────────────────────────────────
def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.video:
        build_parser().print_help()
        sys.exit(1)

    # 1. Najdi LPM SDK a importni bindingy.
    sdk_path, lpm_lib, modules_dir, view_config, wrappers_dir = find_lpm_sdk(
        args.lpm_sdk)
    if str(wrappers_dir) not in sys.path:
        sys.path.insert(0, str(wrappers_dir))
    from lpm import LPM, LpmBoundingBox, LpmModuleConfig  # noqa: E402
    from er import ER  # noqa: E402
    log.info("LPM SDK: %s", sdk_path)
    log.info("LPM lib: %s", lpm_lib.name)

    # Support libs (libopenblas.dll).
    lib_suffixes = (".dll", ".so", ".dylib")
    libs_dir = lpm_lib.parent
    support_libs = [str(p) for p in libs_dir.iterdir()
                    if p.suffix.lower() in lib_suffixes
                    and "lpm" not in p.name.lower()]
    if support_libs:
        log.info("LPM support libs: %s",
                 ", ".join(Path(p).name for p in support_libs))

    ffi = FFI()
    eyedea_er = ER(ffi, str(lpm_lib), support_libs)
    lpm = LPM(ffi, str(lpm_lib), support_libs)
    lpm.init_lpm(str(modules_dir))
    log.info("LPM initialized — version %s", lpm.get_version())

    view = lpm.load_view_config(str(view_config))
    mod_cfg = LpmModuleConfig()
    mod_cfg.ocr_compute_on_gpu = False
    mod_cfg.ocr_num_threads = 1
    mod_cfg.det_compute_on_gpu = False
    mod_cfg.det_num_threads = 1

    module_index = lpm.get_module_index(args.module_id, 0, 0)
    if module_index < 0:
        raise RuntimeError(f"Module ID {args.module_id} not found in {modules_dir}")
    info = lpm.get_module_info(module_index)
    log.info("Module: %s", info.name)
    lpm.load_module(module_index, view, mod_cfg)

    # 2. Otevři video.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %dx%d  %.1f fps  %d frames", width, height, fps, total)
    log.info("Output: %s", args.output_dir)
    log.info("Tracker: %s  Blur: %s  min_conf: %.2f",
             args.tracker, args.blur_method, args.min_confidence)

    os.makedirs(args.output_dir, exist_ok=True)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Video has no readable frames")

    # 3. Najdi počáteční cíl (se scan-ahead).
    target_bbox, source, scanned = select_initial_target(
        first_frame, lpm, eyedea_er, LpmBoundingBox, module_index,
        cap=cap, min_conf=args.min_confidence,
        max_scan_frames=args.initial_scan,
    )
    if target_bbox is None:
        raise RuntimeError("No target could be selected (no LPM detections)")
    if scanned > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot re-read first frame after scan")

    x1, y1, x2, y2 = target_bbox
    tracker = make_tracker(args.tracker)
    init_xywh = tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))
    log.info("Target locked (%s) bbox=(%d,%d)-(%d,%d)", source, x1, y1, x2, y2)

    # 4. Volitelný video writer.
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps,
                                       (width, height))
        if not video_writer.isOpened():
            log.warning("Cannot open video writer for %s — frames only",
                        args.save_video)
            video_writer = None

    # 5. Hlavní smyčka.
    last_known_bbox = target_bbox
    tracking_active = True
    consecutive_lost = 0
    frame_idx = 0
    saved = 0

    try:
        while True:
            if frame_idx == 0:
                frame = first_frame.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            frame_idx += 1

            # A. Tracker update.
            if tracking_active:
                ok_t, rect = tracker.update(frame)
                if ok_t and rect is not None:
                    last_known_bbox = (rect[0], rect[1],
                                       rect[0] + rect[2], rect[1] + rect[3])
                    consecutive_lost = 0
                else:
                    consecutive_lost += 1
                    if consecutive_lost >= args.lost_frames:
                        tracking_active = False
                        log.info("[frame %d] track lost — attempting re-acquisition",
                                 frame_idx)

            # B. Periodická re-detekce (drift korekce).
            if tracking_active and (frame_idx % args.redetect_every == 0):
                detections = run_lpm_detections(lpm, eyedea_er, LpmBoundingBox,
                                                module_index, frame,
                                                args.min_confidence)
                if detections:
                    nearest = find_nearest(detections,
                                           center_of(last_known_bbox),
                                           args.search_radius)
                    if nearest is not None:
                        nx1, ny1, nx2, ny2, _ = nearest
                        tracker.reinit(frame,
                                       (nx1, ny1, nx2 - nx1, ny2 - ny1))
                        last_known_bbox = (nx1, ny1, nx2, ny2)

            # C. Re-akvizice po ztrátě tracku.
            if not tracking_active:
                detections = run_lpm_detections(lpm, eyedea_er, LpmBoundingBox,
                                                module_index, frame,
                                                args.min_confidence)
                nearest = find_nearest(detections,
                                       center_of(last_known_bbox),
                                       args.search_radius * 2)
                if nearest is not None:
                    nx1, ny1, nx2, ny2, nconf = nearest
                    tracker.reinit(frame,
                                   (nx1, ny1, nx2 - nx1, ny2 - ny1))
                    last_known_bbox = (nx1, ny1, nx2, ny2)
                    tracking_active = True
                    consecutive_lost = 0
                    log.info("[frame %d] re-acquired target conf=%.2f",
                             frame_idx, nconf)

            # D + E. Anonymizace + bbox.
            _apply_anonymize(frame, last_known_bbox, args,
                             lost_count=consecutive_lost,
                             tracking=tracking_active)

            # F. Ulož výstup.
            out_path = os.path.join(args.output_dir,
                                    f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            if video_writer is not None:
                video_writer.write(frame)

            if not args.no_preview:
                cv2.imshow("video_anonymizer (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Stopped by user.")
                    break

            if frame_idx % 100 == 0:
                log.info("  ... processed %d frames", frame_idx)

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        lpm.close()

    log.info("Done — %d frames written to %s", saved, args.output_dir)
    log.info("Reassemble: ffmpeg -framerate %.0f -i %s/frame_%%05d.jpg -c:v libx264 out.mp4",
             fps, args.output_dir)


# ─── INTERACTIVE WIZARD ───────────────────────────────────────────
def _sanitize_path(s):
    """Ořeže `& '…'` z PowerShell drag-and-drop a okolní uvozovky."""
    s = s.strip()
    if s.startswith("&"):
        s = s[1:].lstrip()
    if (s.startswith("'") and s.endswith("'")) or \
       (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    return s.strip()


def _ask(prompt, default=None, cast=None):
    raw = input(prompt).strip()
    if not raw and default is not None:
        return default
    if cast is None:
        return raw
    try:
        return cast(raw)
    except (TypeError, ValueError):
        print(f"  ! Neplatná hodnota '{raw}', ponechávám default {default}")
        return default


def _ask_yn(prompt, default=True):
    suf = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suf}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "ano", "a")


def _pick(prompt, options, default_idx=0):
    print(prompt)
    for i, (key, desc) in enumerate(options):
        marker = "▶" if i == default_idx else " "
        print(f"  {marker} {i+1}) {key}  —  {desc}")
    raw = input(f"Volba [1-{len(options)}, default {default_idx+1}]: ").strip()
    if not raw:
        return options[default_idx][0]
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    except ValueError:
        for k, _ in options:
            if raw.lower() == k.lower():
                return k
    print(f"  ! Neplatná volba, ponechávám {options[default_idx][0]}")
    return options[default_idx][0]


def _step(title):
    print()
    print(f"── {title} " + "─" * max(0, 50 - len(title)))


def wizard(argv=None):
    """Interaktivní průvodce — posbírá volby a zavolá main(args)."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  VIDEOANONYMIZER — interaktivní průvodce                ║")
    print("║  (kdykoliv Ctrl+C = konec, prázdný Enter = default)      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    defaults = {
        "video": None, "lpm_sdk": None, "output_dir": DEFAULT_OUTPUT_DIR,
        "tracker": DEFAULT_TRACKER, "redetect_every": DEFAULT_REDETECT_EVERY,
        "lost_frames": DEFAULT_LOST_FRAMES,
        "search_radius": DEFAULT_SEARCH_RADIUS, "module_id": DEFAULT_MODULE_ID,
        "min_confidence": DEFAULT_MIN_CONFIDENCE, "initial_scan": 30,
        "blur_method": DEFAULT_BLUR, "pixel_block": DEFAULT_PIXEL_BLOCK,
        "gaussian_ksize": DEFAULT_GAUSSIAN_KSIZE,
        "bbox_expand": DEFAULT_BBOX_EXPAND, "no_blur": False,
        "draw_bbox": False, "no_preview": False, "save_video": None,
        "log_level": "INFO",
    }

    _step("1/7  Vstupní video / obrázek")
    raw = input("  Cesta k souboru (Enter = výchozí bodycam test): ").strip()
    if not raw:
        default = (Path(__file__).resolve().parent.parent
                   / "LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2"
                   / "wrappers" / "python"
                   / "capture of publicly available bodycam footage.mp4")
        if default.is_file():
            defaults["video"] = str(default)
            print(f"  ▶ {default.name}")
        else:
            raw = input("  Cesta k souboru: ").strip()
            defaults["video"] = _sanitize_path(raw)
    else:
        defaults["video"] = _sanitize_path(raw)

    _step("2/7  Tracker")
    trk = _pick("  Jaký tracker?", [
        ("csrt", "CSRT — přesný, pomalejší (výchozí)"),
        ("kcf",  "KCF — rychlejší, méně přesný"),
        ("kalman", "Kalman — predictor, nejpomalejší drift"),
    ])
    defaults["tracker"] = trk

    _step("3/7  Metoda anonymizace")
    blur = _pick("  Jaký blur?", [
        ("pixelate", "Mosaic / pixelizace (výchozí)"),
        ("gaussian", "Gaussovské rozmazání"),
        ("blackout", "Černý obdélník"),
        ("none",     "Žádná změna (jen tracker + bbox)"),
    ])
    defaults["blur_method"] = blur
    if blur == "pixelate":
        defaults["pixel_block"] = _ask(
            "  Velikost pixel-bloku [8-50, default 18]: ",
            default=DEFAULT_PIXEL_BLOCK, cast=int)
    elif blur == "gaussian":
        defaults["gaussian_ksize"] = _ask(
            "  Kernel size (liché) [11-61, default 21]: ",
            default=DEFAULT_GAUSSIAN_KSIZE, cast=int)
    defaults["bbox_expand"] = _ask(
        "  Expanze bboxu (0.0–0.5, default 0.15): ",
        default=DEFAULT_BBOX_EXPAND, cast=float)

    _step("4/7  Detektor (LPM)")
    defaults["min_confidence"] = _ask(
        "  Minimální confidence [0.1-0.9, default 0.40]: ",
        default=DEFAULT_MIN_CONFIDENCE, cast=float)
    defaults["initial_scan"] = _ask(
        "  Scan-ahead při startu (framů) [0-60, default 30]: ",
        default=30, cast=int)

    _step("5/7  Výstup")
    defaults["output_dir"] = _ask(
        "  Adresář pro výstupní framy [default output_video_frames]: ",
        default=DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    print("  Kde uložit finální video?")
    print("    ▶ 1) out.mp4  (v aktuálním adresáři)")
    print("      2) vlastní cesta")
    print("      3) neukládat video, jen framy")
    video_choice = _ask("  Volba [1]: ", default="1")
    if video_choice == "2":
        defaults["save_video"] = _ask(
            "  Cesta k MP4: ", default="out.mp4")
    elif video_choice == "3":
        defaults["save_video"] = None
    else:
        defaults["save_video"] = "out.mp4"

    _step("6/7  Debug / preview")
    defaults["draw_bbox"] = _ask_yn("  Vykreslit zelený bbox?", default=False)
    defaults["no_preview"] = not _ask_yn(
        "  Zobrazit live preview okno?", default=True)

    _step("7/7  Souhrn")
    print(f"  Video:        {defaults['video']}")
    print(f"  Tracker:      {defaults['tracker']}")
    print(f"  Blur:         {defaults['blur_method']}"
          + (f"  block={defaults['pixel_block']}"
             if defaults['blur_method'] == "pixelate" else "")
          + (f"  ksize={defaults['gaussian_ksize']}"
             if defaults['blur_method'] == "gaussian" else ""))
    print(f"  min_conf:     {defaults['min_confidence']:.2f}")
    print(f"  Output dir:   {defaults['output_dir']}")
    print(f"  Video:        {defaults['save_video'] or '(jen framy)'}")
    print(f"  draw_bbox:    {defaults['draw_bbox']}")
    print(f"  preview:      {not defaults['no_preview']}")
    print()
    if not _ask_yn("  Spustit pipeline nyní s tímto nastavením?", default=True):
        print("Zrušeno.")
        return 0

    args = argparse.Namespace(**defaults, wizard=False)
    extra = _build_argv_from_defaults(defaults)
    try:
        main([defaults["video"]] + extra)
    except SystemExit as e:
        return e.code or 0
    return 0


# ─── PŘEBALENÍ DEFAULTS → ARGV (používá wizard) ──────────────────
def _build_argv_from_defaults(d):
    """Převede slovník nastavení na argv list, který projde argparse.

    Skipne None / False (defaulty) a store_true flagy s False. Store_true
    flagy s True přidá bez hodnoty. Ostatní přidá jako --flag value.
    """
    store_true_keys = {"wizard", "no_blur", "draw_bbox", "no_preview"}
    argv = []
    for k, v in d.items():
        if k == "video":
            continue
        if v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if k in store_true_keys:
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    return argv





if __name__ == "__main__":
    if len(sys.argv) <= 1 or "--wizard" in sys.argv:
        sys.exit(wizard())
    main()
