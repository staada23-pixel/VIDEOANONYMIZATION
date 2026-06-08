"""Hlavní CLI vstup video_anonymizer.

Příklad použití (z project root):

    # video → anonymizované video (mp4)
    python -m video_anonymizer.cli run \\
        --input video.mp4 --output-video data/out.mp4 \\
        --tracker csrt --detector lpm --blur-method pixelate

    # jeden obrázek → anonymizovaný obrázek + json metadata
    python -m video_anonymizer.cli run \\
        --input shot.jpg --output-frames data/out_img/ \\
        --output-json data/out_img.json --no-preview

    # adresář snímků → adresář anonymizovaných + mp4
    python -m video_anonymizer.cli run \\
        --input data/raw/ --output-frames data/out_frames/ \\
        --output-video data/out.mp4 --blur-method blackout

    # info o konfiguraci
    python -m video_anonymizer.cli info
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import yaml

from .utils.logging import setup_logging, get_logger
from .utils.overlap_fn import iou, is_likely_face, apply_nms
from .io.video_reader import (detect_input_type, open_input, iter_frames,
                              video_meta, IMAGE_EXTS, VIDEO_EXTS)
from .io.frame_writer import (FrameWriter, VideoWriter, JSONWriter,
                              MultiWriter, make_writer)
from .anonymizer.blur import (anonymize_region, draw_bbox, expand_bbox,
                              METHODS as BLUR_METHODS)
from .tracking.csrt import CSRTTracker
from .tracking.kalman import KalmanBoxTracker
from .tracking.kcf import KCFTracker
from .tracking.iou_tracker import IOUTracker
from .tracking.track import Track
from .detection.detection_model import Detection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_BLUR_CONFIG = PROJECT_ROOT / "configs" / "blur.yaml"


def load_config(path):
    if path is None:
        path = DEFAULT_CONFIG
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_config():
    return {
        "lpm": {"module_id": 802, "version": 7,
                "det_compute_on_gpu": False, "det_num_threads": 1,
                "ocr_compute_on_gpu": False, "ocr_num_threads": 1},
        "detection": {"backend": "lpm", "min_confidence": 0.4,
                      "redetect_every_n": 15, "min_face_ratio": 0.03,
                      "nms_iou_threshold": 0.3},
        "tracker": {"type": "csrt", "lost_timeout": 20},
        "blur": {"method": "pixelate", "block": 18, "ksize": 51,
                 "color": [0, 0, 0], "expand": 0.15},
        "visualisation": {"enabled": True, "show_bbox": True,
                          "show_label": True, "bbox_color": [0, 255, 0],
                          "bbox_thickness": 2, "font_scale": 0.5,
                          "lost_bbox_color": [0, 0, 255]},
        "input": {"type": "auto", "start_frame": 0, "end_frame": None},
        "output": {"frames": "data/output_frames", "video": None,
                   "json": None, "jpg_quality": 90, "video_codec": "mp4v"},
    }


def merge_yaml(base, override):
    """Rekurzivně přepíše base hodnotami z override (None se ignoruje)."""
    if override is None:
        return base
    if not isinstance(override, dict):
        return override
    out = dict(base or {})
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_yaml(out[k], v)
        else:
            out[k] = v
    return out


def make_tracker(tracker_type, frame, bbox, conf):
    tracker_type = tracker_type.lower()
    if tracker_type == "csrt":
        return CSRTTracker(frame, bbox, conf)
    if tracker_type == "kcf":
        return KCFTracker(frame, bbox, conf)
    if tracker_type == "kalman":
        kf = KalmanBoxTracker(bbox)
        kf.update(bbox)
        return kf
    raise ValueError(f"Unknown tracker type: {tracker_type}")


def get_detector(name, config, lpm_lib=None, modules_dir=None, view_config=None):
    name = (name or "lpm").lower()
    if name == "lpm":
        from .detection.lpm_wrapper import LPMDetector
        return LPMDetector(config, lpm_lib=lpm_lib, modules_dir=modules_dir,
                           view_config=view_config)
    if name == "mediapipe":
        from .detection.mediapipe_detector import FaceHandDetector
        return FaceHandDetector(
            min_detection_confidence=config.get("min_confidence", 0.4))
    raise ValueError(f"Unknown detector: {name}")


def cmd_run(args, log):
    cfg = load_config(args.config)
    if args.blur_config:
        blur_cfg = load_config(args.blur_config)
        cfg = merge_yaml(cfg, {"blur": blur_cfg})

    in_cfg = cfg.get("input", {})
    input_type = args.input_type or in_cfg.get("type", "auto")
    if input_type == "auto":
        input_type = detect_input_type(args.input)
        if input_type == "unknown":
            log.error("Cannot detect input type for: %s", args.input)
            return 1

    if input_type == "image":
        out_spec = {"frames": args.output_frames or "data/output_image"}
        if args.output_json:
            out_spec["json"] = args.output_json
    else:
        out_spec = {
            "frames": args.output_frames or cfg.get("output", {}).get("frames"),
            "video": args.output_video or cfg.get("output", {}).get("video"),
            "json": args.output_json or cfg.get("output", {}).get("json"),
        }

    in_type, source = open_input(args.input, input_type=input_type)
    log.info("Input: %s (type=%s)", args.input, in_type)

    fps, frame_size = 30.0, None
    cap = None
    if in_type in ("video", "webcam"):
        cap = source
        meta = video_meta(cap)
        fps = args.fps or meta["fps"]
        frame_size = (meta["width"], meta["height"])
        log.info("Video: %dx%d  %.1f fps  %d frames",
                 meta["width"], meta["height"], fps, meta["total"])
    elif in_type == "image":
        frame_size = (source.shape[1], source.shape[0])
    elif in_type == "image_dir":
        first = cv2.imread(str(source[0]))
        frame_size = (first.shape[1], first.shape[0])
        log.info("Image dir: %d frames, size %dx%d", len(source), *frame_size)

    det_cfg = cfg.get("detection", {})
    trk_cfg = cfg.get("tracker", {})
    blur_cfg = cfg.get("blur", {})
    vis_cfg = cfg.get("visualisation", {})

    tracker_type = args.tracker or trk_cfg.get("type", "csrt")
    detector_name = args.detector or det_cfg.get("backend", "lpm")
    redetect_every = (args.redetect_every
                      or det_cfg.get("redetect_every_n", 15))
    min_conf = det_cfg.get("min_confidence", 0.4)
    min_face_ratio = det_cfg.get("min_face_ratio", 0.03)

    blur_method = args.blur_method or blur_cfg.get("method", "pixelate")
    blur_block = blur_cfg.get("block", 18)
    blur_ksize = blur_cfg.get("ksize", 51)
    blur_color = tuple(blur_cfg.get("color", [0, 0, 0]))
    blur_expand = blur_cfg.get("expand", 0.15)
    do_anonymize = not args.no_anonymize and blur_method != "none"

    lost_timeout = trk_cfg.get("lost_timeout", 20)
    show_bbox = (vis_cfg.get("show_bbox", True)
                 and vis_cfg.get("enabled", True)
                 and not args.no_boxes)
    bbox_color = tuple(vis_cfg.get("bbox_color", [0, 255, 0]))
    lost_color = tuple(vis_cfg.get("lost_bbox_color", [0, 0, 255]))
    show_label = vis_cfg.get("show_label", True)
    font_scale = vis_cfg.get("font_scale", 0.5)
    bbox_thickness = vis_cfg.get("bbox_thickness", 2)

    start_frame = args.start_frame if args.start_frame is not None else in_cfg.get("start_frame", 0)
    end_frame = args.end_frame if args.end_frame is not None else in_cfg.get("end_frame")

    writer = make_writer(
        {k: v for k, v in out_spec.items() if v},
        fps=fps if input_type in ("video", "image_dir") else None,
        frame_size=frame_size,
    )
    if writer is None and out_spec.get("frames") is None:
        out_spec["frames"] = "data/output_frames"
        writer = FrameWriter(out_spec["frames"])
    if isinstance(writer, JSONWriter) or isinstance(writer, MultiWriter):
        writer.set_meta(input=args.input, input_type=in_type,
                        tracker=tracker_type, detector=detector_name,
                        blur_method=blur_method)

    try:
        detector = get_detector(detector_name, det_cfg)
    except Exception as e:
        log.error("Detector '%s' init failed: %s", detector_name, e)
        return 2

    target_bbox, target_conf = None, 0.0
    tracker = None
    consecutive_lost = 0
    frame_idx = 0
    saved = 0

    try:
        for frame in iter_frames(source, in_type):
            frame_idx += 1
            if frame_idx < start_frame + 1:
                continue
            if end_frame is not None and frame_idx > end_frame:
                break

            h, w = frame.shape[:2]
            detections = detector.detect(frame) if detector else []
            detections = [d for d in detections
                          if d.confidence >= min_conf and is_likely_face(
                              (d.x, d.y, d.x + d.w, d.y + d.h), w, h,
                              min_face_ratio)]

            if tracker is None:
                # Zkoušíme najít cíl každých `redetect_every` framů (scan-ahead).
                # Tak funguje i když první frame obličej neobsahuje.
                if detections and (frame_idx == 1
                                   or frame_idx % redetect_every == 0):
                    best = max(detections, key=lambda d: d.confidence)
                    target_bbox = (best.x, best.y, best.x + best.w,
                                   best.y + best.h)
                    target_conf = best.confidence
                    tracker = make_tracker(tracker_type, frame, target_bbox,
                                           target_conf)
                    log.info("[frame %d] target locked conf=%.2f "
                             "tracker=%s bbox=%s",
                             frame_idx, target_conf, tracker_type, target_bbox)
                elif frame_idx % 30 == 0:
                    log.info("[frame %d] hledám cíl… "
                             "(%d kandidátů z detektoru, "
                             "žádný zatím neprošel filtrem)",
                             frame_idx, len(detections))
            else:
                ok = tracker.update(frame)
                if ok:
                    target_bbox = tracker.get_bbox()
                    consecutive_lost = 0
                else:
                    consecutive_lost += 1

                if (frame_idx % redetect_every == 0
                        or consecutive_lost >= lost_timeout):
                    if detections:
                        def near(d):
                            cx, cy = (d.x + d.w // 2, d.y + d.h // 2)
                            tx = (target_bbox[0] + target_bbox[2]) // 2
                            ty = (target_bbox[1] + target_bbox[3]) // 2
                            return -((cx - tx) ** 2 + (cy - ty) ** 2)
                        best = max(detections, key=near)
                        new_bbox = (best.x, best.y, best.x + best.w,
                                    best.y + best.h)
                        if hasattr(tracker, "reinit"):
                            tracker.reinit(frame, new_bbox, best.confidence)
                        else:
                            tracker = make_tracker(tracker_type, frame,
                                                   new_bbox, best.confidence)
                        target_bbox = new_bbox
                        target_conf = best.confidence
                        consecutive_lost = 0

            anon_bbox = None
            if target_bbox is not None and consecutive_lost < lost_timeout:
                if do_anonymize:
                    anonymize_region(frame, target_bbox, method=blur_method,
                                     block=blur_block,
                                     ksize=(blur_ksize, blur_ksize),
                                     color=blur_color, expand=blur_expand)
                    anon_bbox = expand_bbox(target_bbox, blur_expand, frame.shape)
                if show_bbox:
                    color = bbox_color if consecutive_lost == 0 else lost_color
                    label = None
                    if show_label:
                        label = f"id=1 conf={target_conf:.2f}"
                        if consecutive_lost > 0:
                            label += f" lost={consecutive_lost}"
                    draw_bbox(frame, target_bbox, color=color,
                              thickness=bbox_thickness, label=label,
                              font_scale=font_scale)

            if writer is None:
                out_dir = Path(out_spec.get("frames") or "data/output_frames")
                FrameWriter(out_dir).write(frame_idx, frame)
            elif isinstance(writer, (FrameWriter, VideoWriter, MultiWriter)):
                kwargs = {}
                if isinstance(writer, (JSONWriter, MultiWriter)):
                    kwargs = {"detections": detections,
                              "track": Track(id=1, bbox=target_bbox,
                                             confidence=target_conf,
                                             lost_frames=consecutive_lost)
                                       if target_bbox else None,
                              "anonymized_bbox": anon_bbox}
                if isinstance(writer, MultiWriter):
                    writer.write(frame_idx, frame, **kwargs)
                else:
                    writer.write(frame_idx, frame)
            saved += 1

            if not args.no_preview:
                cv2.imshow("video_anonymizer (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Stopped by user.")
                    break

            if frame_idx % 100 == 0:
                log.info("  ... processed %d frames", frame_idx)
    finally:
        if cap is not None:
            cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.close()
        if detector is not None and hasattr(detector, "close"):
            detector.close()

    log.info("Done — %d frames processed", saved)
    if out_spec.get("frames"):
        log.info("Frames: %s", out_spec["frames"])
    if out_spec.get("video"):
        log.info("Video:  %s", out_spec["video"])
    if out_spec.get("json"):
        log.info("JSON:   %s", out_spec["json"])
    if out_spec.get("video"):
        log.info("ffmpeg reassemble (pokud bys chtěl ručně): "
                 "ffmpeg -framerate %.0f -i %s/frame_%%06d.jpg -c:v libx264 out.mp4",
                 fps, out_spec["frames"])
    return 0


def cmd_info(args, log):
    cfg = load_config(args.config)
    log.info("Loaded config from: %s", args.config or DEFAULT_CONFIG)
    for key in ("lpm", "detection", "tracker", "blur", "visualisation",
                "input", "output"):
        log.info("%-14s %s", key + ":", cfg.get(key))
    return 0


def cmd_blur_info(args, log):
    log.info("Blur methods: %s", ", ".join(BLUR_METHODS))
    return 0


# ─── INTERACTIVE WIZARD ──────────────────────────────────────────

def _ask(prompt, default=None, choices=None, cast=str, allow_empty=False):
    """Interaktivní prompt s defaultem a volitelnou validací choices."""
    casted_default = None
    if default is not None and cast is not bool:
        try:
            casted_default = cast(default)
        except (ValueError, TypeError):
            casted_default = default
    while True:
        suffix = f" [{default}]" if default is not None else ""
        if choices:
            suffix += f"  ({'/'.join(choices)})"
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            return casted_default
        if not raw:
            if casted_default is not None or allow_empty:
                return casted_default
            if default is not None:
                return default
            print("  (povinné, nelze nechat prázdné)")
            continue
        if choices and raw.lower() not in [c.lower() for c in choices]:
            print(f"  Neplatná volba. Zadej jedno z: {', '.join(choices)}")
            continue
        if cast is bool:
            v = raw.lower()
            if v in ("y", "yes", "ano", "a", "1", "true"):
                return True
            if v in ("n", "no", "ne", "0", "false"):
                return False
            print("  Zadej y/n")
            continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"  Neplatná hodnota, zkus to znovu.")
            continue


def _yn(prompt, default=True):
    return _ask(prompt, default="y" if default else "n", cast=bool)


def _step(n, total, title):
    """Vypíše hlavičku kroku s progressem."""
    print()
    print(f"  ┌─ Krok {n}/{total}  {title}")


def _end_step():
    print("  └─")


def _pick(title, options, default_key=None):
    """Zobrazí číslovaný seznam voleb s popisem a vrátí vybraný klíč.

    options: list of (key, label, description)  nebo  list of (key, label)
    """
    print(f"  │ {title}")
    print("  │")
    for i, opt in enumerate(options, 1):
        if len(opt) == 3:
            key, label, desc = opt
        else:
            key, label = opt
            desc = ""
        marker = "  " if (default_key and key != default_key) else (
            "→ " if default_key and key == default_key else "  ")
        if default_key and key == default_key:
            marker = "▶ "
        line = f"  │   {marker}{i}) {label}"
        if desc:
            line += f"   — {desc}"
        print(line)
    print("  │")
    if default_key:
        default_idx = next((i for i, o in enumerate(options, 1)
                            if o[0] == default_key), 1)
        prompt_suffix = f" [{default_idx}]"
    else:
        prompt_suffix = ""
    while True:
        try:
            raw = input(f"  │ Volba{prompt_suffix}: ").strip()
        except EOFError:
            return default_key
        if not raw and default_key:
            return default_key
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
            print(f"  │   Neplatné číslo (1–{len(options)})")
            continue
        # příjme i název (case-insensitive)
        for opt in options:
            if raw.lower() == opt[0].lower():
                return opt[0]
        print(f"  │   Neplatná volba, zadej číslo nebo název")


def _ask_path(title, hint, allow_picker=True, allow_empty=False, default=""):
    """Prompt na cestu s podporou drag&drop, file pickeru a prázdného vstupu."""
    print(f"  │ {title}")
    print(f"  │ {hint}")
    if allow_picker:
        print("  │ Tip: napiš 'p' pro nativní file dialog (vyžaduje tkinter).")
    print("  │")
    while True:
        try:
            raw = input(f"  │ Cesta [{default or 'prázdné = přeskočit'}]: ").strip()
        except EOFError:
            return None
        if not raw:
            if default:
                return default
            if allow_empty:
                return None
            print("  │   (prázdné není povolené, zadej cestu nebo 'p' pro picker)")
            continue
        if allow_picker and raw.lower() == "p":
            picked = _maybe_pick_file()
            if picked:
                print(f"  │   vybráno: {picked}")
                return _sanitize_path(picked)
            print("  │   file picker nedostupný, zadej cestu ručně")
            continue
        return _sanitize_path(raw)


def _ask_yn(title, default=True):
    """Yes/no prompt s title."""
    print(f"  │ {title}")
    yn = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            raw = input(f"  │ {yn}: ").strip().lower()
        except EOFError:
            return default
        if not raw:
            return default
        if raw in ("y", "yes", "ano", "a", "1"):
            return True
        if raw in ("n", "no", "ne", "0"):
            return False
        print("  │   Zadej y nebo n")


def _ask_int(title, hint, default, min_val=None, max_val=None):
    """Int prompt s title a hintem."""
    print(f"  │ {title}")
    print(f"  │ {hint}")
    while True:
        try:
            raw = input(f"  │ Hodnota [{default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  │   Zadej celé číslo")
            continue
        if min_val is not None and v < min_val:
            print(f"  │   Minimum je {min_val}")
            continue
        if max_val is not None and v > max_val:
            print(f"  │   Maximum je {max_val}")
            continue
        return v


def _print_banner():
    print()
    print("=" * 60)
    print("  Video Anonymizer — Interactive Setup")
    print("=" * 60)
    print()
    print("  Na každý dotaz můžeš stisknout Enter pro defaultní volbu.")
    print("  Výběr z menu: napiš číslo (1, 2, 3…) nebo název volby.")
    print()


def _print_summary(spec):
    print()
    print("=" * 60)
    print("  Souhrn nastavení")
    print("=" * 60)
    rows = [
        ("Vstup",          spec.get("input") or "(webcam)"),
        ("Typ vstupu",     spec.get("input_type") or "auto"),
        ("Detektor",       spec.get("detector") or "lpm"),
        ("Tracker",        spec.get("tracker") or "csrt"),
        ("Blur",           spec.get("blur_method") or "pixelate"),
        ("Výstup (frames)", spec.get("output_frames") or "data/output_frames"),
        ("Výstup (video)",  spec.get("output_video") or "(přeskočit)"),
        ("Výstup (json)",   spec.get("output_json") or "(přeskočit)"),
        ("Re-detekce",      f"každých {spec.get('redetect_every') or 15} framů"),
        ("Preview okno",    "ne" if spec.get("no_preview") else "ano"),
        ("Bounding boxy",   "ne" if spec.get("no_boxes") else "ano"),
        ("Start frame",     spec.get("start_frame") or 0),
        ("End frame",       spec.get("end_frame") or "(do konce)"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}} : {v}")
    print("=" * 60)


def _build_args_from_spec(spec, base_args):
    """Z dictu spec vytvoří Namespace compatibilní s cmd_run."""
    import argparse
    import copy
    args = copy.copy(base_args)
    # Fallback na všechny atributy, které cmd_run čte, ale `interactive` subparser
    # je nemá. Tím zajistíme, že args.blur_config apod. vždy existuje.
    fallback = {
        "blur_config": None,
        "fps": None,
        "no_anonymize": False,
        "no_preview": False,
        "no_boxes": False,
        "start_frame": None,
        "end_frame": None,
        "redetect_every": None,
    }
    for k, v in fallback.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    for k, v in spec.items():
        setattr(args, k, v)
    return args


def _sanitize_path(raw):
    """Očistí cestu od PowerShell drag-and-drop artefaktů (`& '...'`)."""
    if raw is None:
        return raw
    s = raw.strip()
    if s.startswith("& "):
        s = s[2:].strip()
    if (s.startswith("'") and s.endswith("'")) or (
            s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    return s


def _maybe_pick_file():
    """Pokusí se otevřít nativní file dialog (tkinter). Vrátí cestu nebo None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Vyber video nebo obrázek",
            filetypes=[
                ("Media", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                ("Video", "*.mp4 *.avi *.mov *.mkv"),
                ("Obrázky", "*.jpg *.jpeg *.png *.bmp"),
                ("Vše", "*.*"),
            ],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


# Definice voleb s popisem
INPUT_TYPE_OPTS = [
    ("auto",      "auto-detect",  "podle přípony souboru (doporučeno)"),
    ("video",     "video soubor", ".mp4 / .avi / .mov / .mkv"),
    ("image",     "jeden obrázek", ".jpg / .png / .bmp"),
    ("image_dir", "adresář obrázků", "sekvence, tříděná abecedně"),
    ("webcam",    "webkamera",    "živý stream z kamery"),
]

DETECTOR_OPTS = [
    ("lpm",       "LPM SDK",      "proprietární Eyedea, vyžaduje HASP + licenci"),
    ("mediapipe", "MediaPipe",    "open-source od Googlu, `pip install mediapipe`"),
]

TRACKER_OPTS = [
    ("csrt",   "CSRT",   "přesný, pomalejší — vhodný pro bodycam"),
    ("kcf",    "KCF",    "rychlejší, méně přesný při rotaci/škálování"),
    ("kalman", "Kalman", "nejrychlejší, predikuje pohyb (žádný vizuální model)"),
]

BLUR_OPTS = [
    ("pixelate", "Pixelate (mozaika)", "dlaždice, silná anonymizace"),
    ("gaussian", "Gaussian blur",      "rozmazání přes ROI"),
    ("blackout", "Blackout (černý box)", "plný barevný obdélník"),
    ("none",     "Žádný (jen bbox)",   "pouze vizualizace, beze změny obrazu"),
]


def cmd_interactive(args, log):
    TOTAL = 13
    _print_banner()
    spec = {}

    # 1) Vstup
    _step(1, TOTAL, "Vstup")
    inp = _ask_path(
        title="Cesta k videu / obrázku / adresáři.",
        hint="Prázdné = webcam. Můžeš přetáhnout soubor myší do okna terminálu.",
        allow_picker=True,
        default="",
    )
    _end_step()
    if inp:
        spec["input"] = inp
    else:
        spec["input"] = None
        spec["input_type"] = "webcam"

    # 2) Typ vstupu
    if "input_type" not in spec:
        _step(2, TOTAL, "Typ vstupu")
        spec["input_type"] = _pick(
            "Jaký typ média je tvůj vstup?",
            INPUT_TYPE_OPTS, default_key="auto")
        _end_step()

    # 3) Detektor
    _step(3, TOTAL, "Detektor")
    spec["detector"] = _pick(
        "Který detektor obličejů použít?",
        DETECTOR_OPTS, default_key="lpm")
    _end_step()

    # 4) Tracker
    _step(4, TOTAL, "Tracker")
    spec["tracker"] = _pick(
        "Který tracker použít po počáteční detekci?",
        TRACKER_OPTS, default_key="csrt")
    _end_step()

    # 5) Blur
    _step(5, TOTAL, "Anonymizační metoda")
    spec["blur_method"] = _pick(
        "Jakým způsobem anonymizovat rozpoznaný obličej?",
        BLUR_OPTS, default_key="pixelate")
    _end_step()

    # 6) Výstup frames
    _step(6, TOTAL, "Výstup — adresář pro JPEG snímky")
    spec["output_frames"] = _ask_path(
        title="Kam ukládat jednotlivé anonymizované snímky?",
        hint="Adresář bude vytvořen, pokud neexistuje.",
        allow_picker=False, default="data/output_frames")
    _end_step()

    # 7) Výstup video
    _step(7, TOTAL, "Výstup — video soubor (.mp4)")
    out_vid = _ask_path(
        title="Volitelné. Kam uložit výsledné anonymizované video?",
        hint="Prázdné = přeskočit (uloží se jen snímky).",
        allow_picker=False, default="", allow_empty=True)
    _end_step()
    spec["output_video"] = out_vid if out_vid else None

    # 8) Výstup JSON
    _step(8, TOTAL, "Výstup — JSON metadata")
    out_json = _ask_path(
        title="Volitelné. Kam uložit per-frame JSON (detekce + track)?",
        hint="Prázdné = přeskočit.",
        allow_picker=False, default="", allow_empty=True)
    _end_step()
    spec["output_json"] = out_json if out_json else None

    # 9) Re-detekce
    _step(9, TOTAL, "Re-detekce (drift korekce)")
    spec["redetect_every"] = _ask_int(
        title="Jak často (každých N framů) znovu spustit detektor?",
        hint="Menší hodnota = přesnější, ale pomalejší. Doporučeno 10–20.",
        default=15, min_val=1, max_val=1000)
    _end_step()

    # 10) Preview
    _step(10, TOTAL, "Preview okno")
    spec["no_preview"] = not _ask_yn(
        "Zobrazit live náhledové okno během zpracování?", default=False)
    _end_step()

    # 11) Bbox
    _step(11, TOTAL, "Bounding boxy")
    spec["no_boxes"] = not _ask_yn(
        "Kreslit bounding boxy (zelený obdélník + label) do výstupu?",
        default=True)
    _end_step()

    # 12) Start frame
    _step(12, TOTAL, "Start frame")
    sf = _ask_int(
        title="Od kterého framu (0-indexed) začít?",
        hint="0 = od začátku. Vhodné pro oříznutí úvodu videa.",
        default=0, min_val=0, max_val=10**7)
    _end_step()
    spec["start_frame"] = sf if sf > 0 else None

    # 13) End frame
    _step(13, TOTAL, "End frame")
    ef = _ask_int(
        title="Na kterém framu (inclusive) skončit?",
        hint="Prázdné / 0 = do konce videa.",
        default=0, min_val=0, max_val=10**7)
    _end_step()
    spec["end_frame"] = ef if ef > 0 else None

    _print_summary(spec)
    print()
    if not _ask_yn("Spustit pipeline nyní s tímto nastavením?", default=True):
        print("Zrušeno.")
        return 0
    print()

    run_args = _build_args_from_spec(spec, args)
    return cmd_run(run_args, log)


def build_parser():
    p = argparse.ArgumentParser(
        prog="video_anonymizer",
        description=("CLI pro video anonymization pipeline "
                     "(LPM/MediaPipe + CSRT/KCF/Kalman + pixelate/gaussian/blackout)."),
    )
    p.add_argument("--config", "-c", default=None,
                   help=f"Cesta k YAML configu (default: {DEFAULT_CONFIG})")
    p.add_argument("--log-level", default="INFO",
                   help="DEBUG/INFO/WARNING/ERROR")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Spustí anonymizační pipeline")

    p_run.add_argument("--input", "-i", default=None,
                       help="Vstup: video soubor / obrázek / adresář / (prázdné=webcam)")
    p_run.add_argument("--input-type", default=None,
                       choices=["auto", "video", "image", "image_dir", "webcam"],
                       help="Typ vstupu (default: auto-detect)")

    p_run.add_argument("--output-frames", default=None,
                       help="Adresář pro sekvenci JPEG snímků")
    p_run.add_argument("--output-video", default=None,
                       help="Cesta k výstupnímu video souboru (.mp4/.avi)")
    p_run.add_argument("--output-json", default=None,
                       help="Cesta k JSON metadatům (per-frame detekce/track)")
    p_run.add_argument("--fps", type=float, default=None,
                       help="FPS výstupního videa (default: z video meta)")

    p_run.add_argument("--tracker", "-t", default=None,
                       choices=["csrt", "kcf", "kalman"],
                       help="Typ trackeru (přepíše config)")
    p_run.add_argument("--detector", "-d", default=None,
                       choices=["lpm", "mediapipe"],
                       help="Typ detektoru (přepíše config)")
    p_run.add_argument("--redetect-every", type=int, default=None,
                       help="Re-detekce každých N framů (přepíše config)")

    p_run.add_argument("--blur-method", "-b", default=None,
                       choices=list(BLUR_METHODS),
                       help="Anonymizační metoda (přepíše config)")
    p_run.add_argument("--blur-config", default=None,
                       help=f"Cesta k blur YAML (default: {DEFAULT_BLUR_CONFIG})")
    p_run.add_argument("--no-anonymize", action="store_true",
                       help="Vypne anonymizaci (ponechá jen bbox overlay)")

    p_run.add_argument("--no-preview", action="store_true",
                       help="Vypne live preview okno")
    p_run.add_argument("--no-boxes", action="store_true",
                       help="Vypne vykreslování bounding boxů")

    p_run.add_argument("--start-frame", type=int, default=None,
                       help="První frame ke zpracování (0-indexed)")
    p_run.add_argument("--end-frame", type=int, default=None,
                       help="Poslední frame ke zpracování (inclusive)")
    p_run.set_defaults(func=cmd_run)

    p_info = sub.add_parser("info", help="Vypíše aktuální konfiguraci")
    p_info.set_defaults(func=cmd_info)

    p_bi = sub.add_parser("blur-info",
                          help="Vypíše dostupné anonymizační metody")
    p_bi.set_defaults(func=cmd_blur_info)

    p_i = sub.add_parser("interactive", aliases=["i", "wizard"],
                         help="Interaktivní průvodce (krok za krokem)")
    p_i.set_defaults(func=cmd_interactive)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    log = setup_logging(args.log_level)
    return args.func(args, log)


if __name__ == "__main__":
    sys.exit(main())
