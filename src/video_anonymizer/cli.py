"""
CLI entry point pro video anonymizer.

Rezimy:
  python -m video_anonymizer --interactive      # TUI menu
  python -m video_anonymizer --input ...        # rychly, bez menu
  python -m video_anonymizer                    # auto-detekce: kdyz chybi --input, spusti TUI
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from video_anonymizer.io.video_reader import VideoReader
from video_anonymizer.io.frame_writer import FrameWriter
from video_anonymizer.detection.lpm_wrapper import LPMWrapper
from video_anonymizer.detection.hog_detector import HOGDetector
from video_anonymizer.detection.face_detector import FaceDetector
from video_anonymizer.tracking.byte_tracker import ByteTracker
from video_anonymizer.tracking.kcf import KCFTracker
from video_anonymizer.tracking.opencv_tracker import OpenCVTracker
from video_anonymizer.tracking.vit_tracker import ViTTracker
from video_anonymizer.tracking.structures import TrackState
from video_anonymizer.utils.cmc import CameraMotionCompensator
from video_anonymizer.utils.anonymizer import Anonymizer
from video_anonymizer.utils.logging_utils import (
    print_startup_info,
    format_frame_info,
    print_final_stats,
)
from video_anonymizer.utils.frame_history import FrameHistory, TrackFrame
from video_anonymizer.utils.backfill import BackwardFiller
from video_anonymizer import tui


# ── Config loading ──────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(path: str) -> str:
    """
    Najdi config.yaml. Zkousi:
      1. absolutni cesta (jak je)
      2. relativni k CWD
      3. relativni k _PROJECT_ROOT (= project/)
    Pokud nic neexistuje, vrati puvodni `path` (open pak vyhodi FileNotFoundError
    se srozumitelnou hlaskou).
    """
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)
    cand = os.path.join(_PROJECT_ROOT, path)
    if os.path.isfile(cand):
        return cand
    return path  # at to spadne s FileNotFoundError nize


# ── Tracker factory ─────────────────────────────────────────────

def _make_tracker_factory(tracker_name: str, tracker_config: dict):
    name = tracker_name.upper()
    if name == "KCF":
        return lambda: KCFTracker(tracker_config)
    if name in ("CSRT", "MIL"):
        return lambda: OpenCVTracker(name, tracker_config)
    if name == "VIT":
        return lambda: ViTTracker(tracker_config)
    raise ValueError(f"Neznámý tracker: {tracker_name!r}")


# ── Detector factory ────────────────────────────────────────────

def _build_detector(kind: str, main_cfg: dict):
    """`auto` zkusi LPM, pak spadne na face DNN."""
    if kind in ("face", "yunet"):
        print("[detector] Face DNN (YuNet) — obličeje, bez HASP")
        return FaceDetector(main_cfg.get("detector", {}))

    if kind == "hog":
        print("[detector] HOG (OpenCV) — celé postavy")
        return HOGDetector(main_cfg.get("detector", {}))

    if kind == "haar":
        print("[detector] Haar Cascade — obličeje (fallback)")
        return FaceDetector({"confidence": 0.5})

    if kind == "lpm":
        print("[detector] LPM (Eyedea) — vyžaduje HASP")
        return LPMWrapper(main_cfg)

    # auto
    try:
        print("[detector] Pokouším se inicializovat LPM...")
        det = LPMWrapper(main_cfg)
        print("[detector] OK: LPM funguje")
        return det
    except Exception as e:
        print(f"[detector] LPM nedostupný ({type(e).__name__}: {e})")
        print("[detector] → Fallback na Face DNN YuNet")
        return FaceDetector(main_cfg.get("detector", {}))


# ── Vizualizace ─────────────────────────────────────────────────

def _draw_tracks(frame, tracks, viz_cfg, frame_idx, cam_dx, cam_dy, show_boxes: bool):
    if not show_boxes:
        return frame
    annotated = frame.copy()
    lw_auto = max(2, annotated.shape[1] // 400)
    lw = int(viz_cfg.get("box_thickness") or lw_auto)
    font_scale = float(viz_cfg.get("font_scale", 0.5))
    colors_cfg = viz_cfg.get("colors", {})

    def _c(name, default):
        return tuple(int(x) for x in colors_cfg.get(name, default))

    for tr in tracks:
        box = tr.box
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = annotated.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if tr.state == TrackState.ACTIVE:
            color = _c("active", [0, 0, 255])
            label = f"ID{tr.id} {tr.conf:.2f}"
        elif tr.state == TrackState.LOW:
            color = _c("low", [0, 200, 255])
            label = f"ID{tr.id} ~{tr.conf:.2f}"
        elif tr.kcf_ok:
            color = _c("kcf_ok", [0, 140, 255])
            label = f"ID{tr.id}[KCF]"
        else:
            color = _c("waiting", [140, 40, 140])
            label = f"ID{tr.id}[TM:{tr.kcf_template_score:.2f}]"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_y = y1 - 6 if y1 - 6 > th else y2 + th + 6
        cv2.rectangle(annotated, (x1, label_y - th - 3), (x1 + tw + 6, label_y + 3), color, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    if viz_cfg.get("show_info_bar", True):
        info = format_frame_info(frame_idx, tracks, cam_dx, cam_dy)
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return annotated


# ── Pipeline build ──────────────────────────────────────────────

def _build_pipeline(args, main_cfg=None, tracker_cfg=None, viz_cfg=None):
    # Resi config cestu: musi vest na existujici soubor
    cfg_path = _resolve_config_path(args.config)
    args.config = cfg_path

    if main_cfg is None:
        main_cfg = _load_yaml(cfg_path)
    # Vsechny relativni cesty (trackers, viz) vzdy vzhledem k _PROJECT_ROOT,
    # NE vzhledem k main_cfg_dir (= configs/) - tam by hledani selhalo.
    cfg_base = _PROJECT_ROOT

    if tracker_cfg is None:
        if getattr(args, "tracker_config", None):
            tcfg_path = args.tracker_config
        else:
            tcfg_path = os.path.join(
                cfg_base, "configs", "trackers", f"{args.tracker.lower()}.yaml"
            )
        tracker_cfg = _load_yaml(tcfg_path) if os.path.exists(tcfg_path) else {}

    if viz_cfg is None:
        viz_cfg_path = os.path.join(cfg_base, "configs", "trackers", "visualisation.yaml")
        viz_cfg = _load_yaml(viz_cfg_path) if os.path.exists(viz_cfg_path) else {}

    # Webcam = int, jinak cesta
    inp = args.input
    if isinstance(inp, str) and inp.lstrip("-").isdigit():
        source = int(inp)
    else:
        source = inp
    reader = VideoReader(source)
    detector = _build_detector(args.detector, main_cfg)
    cmc_cfg = {
        "cmc_scale": tracker_cfg.get("cmc_scale", 0.4),
        "cmc_pyramid_scale": 0.5,
        "cmc_levels": tracker_cfg.get("cmc_levels", 3),
        "cmc_winsize": tracker_cfg.get("cmc_winsize", 21),
        "cmc_iterations": tracker_cfg.get("cmc_iterations", 3),
        "cmc_poly_n": 5,
        "cmc_poly_sigma": 1.2,
    }
    cmc = CameraMotionCompensator(cmc_cfg) if tracker_cfg.get("cmc_enabled", True) else None
    factory = _make_tracker_factory(args.tracker, tracker_cfg)
    bt = ByteTracker(factory, tracker_cfg)

    anonymizer = Anonymizer(
        method=getattr(args, "anon_method", "mosaic"),
        strength=getattr(args, "anon_strength", 15),
    )

    # Frame writer
    writer = None
    if getattr(args, "out_frames", ""):
        out_cfg = main_cfg.get("output", {}) or {}
        writer = FrameWriter(
            output_dir=args.out_frames,
            save_only_with_detections=(not getattr(args, "save_all", False))
                                       and bool(out_cfg.get("save_only_with_detections", False)),
            save_every_n=int(out_cfg.get("save_every_n_frames", 1)),
        )

    # Video writer
    video_writer = None
    if getattr(args, "out_video", ""):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.out_video, fourcc, reader.fps if reader.fps > 0 else 20.0,
            (reader.width, reader.height),
        )
        if not video_writer.isOpened():
            print(f"[WARN] Nelze otevrit video writer: {args.out_video}")
            video_writer = None

    return reader, detector, cmc, bt, anonymizer, writer, video_writer, viz_cfg, tracker_cfg


# ── Main loop ───────────────────────────────────────────────────

def _run(args) -> int:
    # _build_pipeline si samo vyresi cestu k configu
    reader, detector, cmc, bt, anonymizer, writer, video_writer, viz_cfg, tracker_cfg = \
        _build_pipeline(args)

    print_startup_info(
        reader,
        tracker_name=args.tracker,
        kcf_info={
            "sigma": tracker_cfg.get("sigma"),
            "lambda": tracker_cfg.get("lambda"),
            "learning_rate": tracker_cfg.get("learning_rate"),
            "padding": tracker_cfg.get("padding"),
        },
    )
    print(f"  Detektor:      {type(detector).__name__}")
    print(f"  Anonymizace:   {anonymizer.method}"
          + (f" (sila={anonymizer.strength})" if anonymizer.method != "none" else ""))
    if video_writer:
        print(f"  Video out:     {args.out_video}")
    if writer:
        print(f"  Frames out:    {args.out_frames}")

    show_boxes = not getattr(args, "no_boxes", False)
    show_window = not args.no_display
    is_webcam = reader.is_webcam
    video_width = reader.width
    video_height = reader.height
    do_backfill = getattr(args, "backfill", False) and not is_webcam

    history: FrameHistory | None = None
    if do_backfill:
        history = FrameHistory()
        print("  Backfill:      ANO (doplním pozdní detekce zpětně)")

    start = time.time()
    frame_idx = 0
    saved_frames = 0
    saved_video_frames = 0

    try:
        for frame_idx, frame in reader:
            if getattr(args, "max_frames", 0) and frame_idx >= args.max_frames:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cmc is not None:
                cam_dx, cam_dy = cmc.compute(gray)
            else:
                cam_dx, cam_dy = 0.0, 0.0

            detections = detector.detect(frame)
            tracks = bt.update(detections, frame, cam_dx=cam_dx, cam_dy=cam_dy)

            if history is not None:
                history.record(frame_idx, tracks, cam_dx, cam_dy, len(detections))

            # 1. Anonymizace na cistem snimku
            anon_frame = frame.copy()
            for tr in tracks:
                anonymizer.apply(anon_frame, tr.box)

            # 2. Vykresleni boxu (na anonymizovanem snimku, pokud jsou videt)
            annotated = _draw_tracks(
                anon_frame, tracks, viz_cfg, frame_idx, cam_dx, cam_dy, show_boxes,
            )

            # 3. Uloz
            if writer is not None:
                has_any = len(tracks) > 0
                if writer.write(frame_idx, annotated, has_detections=has_any):
                    saved_frames += 1
            if video_writer is not None:
                video_writer.write(annotated)
                saved_video_frames += 1

            # 4. Zobraz
            if show_window:
                cv2.imshow("Video Anonymizer", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUkončeno uživatelem.")
                    break
    except KeyboardInterrupt:
        print("\nPřerušeno (Ctrl-C).")
    finally:
        elapsed = time.time() - start
        reader.release()
        if show_window:
            cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        detector.close()

    output_path = getattr(args, "out_frames", "") or getattr(args, "out_video", "")
    print_final_stats(frame_idx, saved_frames if writer else saved_video_frames, elapsed, output_path)

    # ── Backward fill ──────────────────────────────────────────────
    if history is not None and len(history.snapshots) > 0:
        filler = BackwardFiller(video_width, video_height)
        backfilled_cnt, modified_idx = filler.fill(history.snapshots)

        if backfilled_cnt > 0:
            has_output = bool(getattr(args, "out_frames", "") or getattr(args, "out_video", ""))
            if has_output:
                print(f"[backfill] Doplňuji {backfilled_cnt} pozdních detekcí zpětně...")
                v_frames, jpg_frames = _re_render(args, history, anonymizer, viz_cfg, show_boxes)
                parts = []
                if v_frames:
                    parts.append(f"{v_frames} framů ve videu")
                if jpg_frames:
                    parts.append(f"{jpg_frames} JPG snímků")
                print(f"[backfill] Hotovo: {' + '.join(parts)} s doplněnými detekcemi")

    return 0

    output_path = getattr(args, "out_frames", "") or getattr(args, "out_video", "")
    print_final_stats(frame_idx, saved_frames if writer else saved_video_frames, elapsed, output_path)

    return 0


# ── Re-render after backfill ────────────────────────────────────

def _re_render(args, history, anonymizer, viz_cfg, show_boxes) -> tuple[int, int]:
    """Re-render output using combined (original + backfilled) tracks.
    Returns (video_frames_written, jpg_frames_saved)."""
    inp = args.input
    if isinstance(inp, str) and inp.lstrip("-").isdigit():
        source = int(inp)
    else:
        source = inp
    reader = VideoReader(source)

    fw = None
    if getattr(args, "out_frames", ""):
        fw = FrameWriter(args.out_frames, save_only_with_detections=False, save_every_n=1)
    vw = None
    if getattr(args, "out_video", ""):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(
            args.out_video, fourcc,
            reader.fps if reader.fps > 0 else 20.0,
            (reader.width, reader.height),
        )
        if not vw.isOpened():
            print(f"[backfill] VAROVÁNÍ: nejde otevřít video writer pro přepis")
            vw = None

    frame_map: dict[int, list[TrackFrame]] = {}
    for snap in history.snapshots:
        frame_map[snap.frame_idx] = snap.tracks

    class _RenderTrack:
        __slots__ = ("box", "id", "conf", "state", "kcf_ok", "kcf_template_score")
        def __init__(self, tf: TrackFrame):
            self.box = tf.box
            self.id = tf.id
            self.conf = tf.conf
            if tf.state == "active":
                self.state = TrackState.ACTIVE
            elif tf.state == "low":
                self.state = TrackState.LOW
            else:
                self.state = TrackState.LOST
            self.kcf_ok = True
            self.kcf_template_score = 0.0

    max_frames = getattr(args, "max_frames", 0)
    total_frames = min(len(reader), max_frames) if max_frames and len(reader) else (len(reader) or max_frames)
    v_frames = 0
    jpg_frames = 0
    step = max(1, total_frames // 20) if total_frames else 50

    for frame_idx, frame in reader:
        if max_frames and frame_idx > max_frames:
            break
        tf_list = frame_map.get(frame_idx, [])
        tracks = [_RenderTrack(tf) for tf in tf_list]

        anon_frame = frame.copy()
        for tr in tracks:
            anonymizer.apply(anon_frame, tr.box)

        annotated = _draw_tracks(anon_frame, tracks, viz_cfg, frame_idx, 0.0, 0.0, show_boxes)

        if vw is not None:
            vw.write(annotated)
            v_frames += 1
        if fw is not None:
            if fw.write(frame_idx, annotated, has_detections=len(tracks) > 0):
                jpg_frames += 1

        if frame_idx % step == 0:
            print(f"  [backfill] přepis... {frame_idx}/{total_frames}")

    reader.release()
    if vw is not None:
        vw.release()
    return v_frames, jpg_frames


# ── Argument parsing ────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_anonymizer",
        description="Detekce + tracking + anonymizace obličejů ve videu.",
    )
    parser.add_argument("--input", default=None,
                        help="Cesta k videu nebo '0' pro webcam. "
                             "Bez tohoto parametru se spustí interaktivní menu.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tracker", default="KCF", choices=["KCF", "CSRT", "MIL", "VIT"])
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--detector", default="auto",
                        choices=["auto", "lpm", "face", "hog", "haar"],
                        help="Detektor: 'face' = YuNet DNN (doporučeno pro bodycam), "
                             "'auto' = zkus LPM, fallback na face")
    parser.add_argument("--anon-method", default="mosaic",
                        choices=["none", "mosaic", "blur", "black", "solid"],
                        help="Způsob anonymizace detekovaných oblastí")
    parser.add_argument("--anon-strength", type=int, default=15,
                        help="Síla efektu (1-50, větší = silnější)")
    parser.add_argument("--no-display", action="store_true",
                        help="Bez okna, jen ukládej")
    parser.add_argument("--no-boxes", action="store_true",
                        help="Bez vykreslování boxů (čistý anonymizovaný obraz)")
    parser.add_argument("--out-video", default="",
                        help="Cesta pro výstupní anonymizované MP4")
    parser.add_argument("--out-frames", default="",
                        help="Složka pro ukládání jednotlivých JPG snímků")
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Zpracuj max N snimku (0 = cely video, pro rychly test nastav napr. 300)")
    parser.add_argument("--interactive", action="store_true",
                        help="Vynutí interaktivní menu")
    parser.add_argument("--backfill", action="store_true", default=False,
                        help="Po forward passu doplní pozdní detekce zpět v čase (inverzní CMC)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Spustí na demo videu s max 50 framy, uloží forward log, pak skončí")
    return parser


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    # ── Dry run ─────────────────────────────────────────────────────
    if args.dry_run:
        print("=" * 60)
        print("  DRY RUN — test pipeline od začátku do konce")
        print("=" * 60)
        import tempfile
        import shutil
        dry_dir = os.path.join(_PROJECT_ROOT, "output", "dry_run")
        if os.path.isdir(dry_dir):
            shutil.rmtree(dry_dir)
        os.makedirs(dry_dir, exist_ok=True)

        demo_video = os.path.join(_PROJECT_ROOT, "tests", "demo.mp4")
        if not os.path.isfile(demo_video):
            print(f"[DRY-RUN] CHYBA: demo video nenalezeno: {demo_video}")
            return 1

        args.input = demo_video
        args.config = "configs/config.yaml"
        args.detector = "face"
        args.tracker = "KCF"
        args.anon_method = "mosaic"
        args.anon_strength = 15
        args.max_frames = 50
        args.backfill = True
        args.out_video = os.path.join(dry_dir, "dry_run_anonymized.mp4")
        args.out_frames = os.path.join(dry_dir, "frames")
        args.no_display = True
        if not hasattr(args, "config") or not args.config:
            args.config = "configs/config.yaml"

        print(f"  Demo video: {demo_video}")
        print(f"  Max framů:  {args.max_frames}")
        print(f"  Výstup:     {dry_dir}")
        print()

        rc = _run(args)

        video_out = args.out_video
        if os.path.isfile(video_out):
            size_mb = os.path.getsize(video_out) / 1024 / 1024
            print(f"[DRY-RUN] Výstupní video: {video_out} ({size_mb:.1f} MB)")
        else:
            print(f"[DRY-RUN] VAROVÁNÍ: výstupní video nenalezeno")

        print(f"[DRY-RUN] Hotovo, kód: {rc}")
        return rc

    # ── Interactive mode ────────────────────────────────────────────
    if args.interactive or args.input is None:
        opts = tui.run_interactive()
        for k, v in opts.items():
            setattr(args, k, v)

    if args.input is None:
        parser.error("--input je povinny (nebo spust bez argumentu pro menu)")

    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
