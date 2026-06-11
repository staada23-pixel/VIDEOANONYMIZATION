"""Pomocné výpisové funkce."""
from __future__ import annotations


def print_startup_info(video_reader, tracker_name: str, kcf_info: dict | None = None) -> None:
    """Vytiskne informace o videu a trackeru pri startu."""
    print(f"\nVideo: {video_reader.source}")
    n_frames = len(video_reader)
    if video_reader.is_webcam:
        frames_str = "(webcam, zpracovava se dokud se neukonci)"
    elif n_frames > 0:
        frames_str = f"{n_frames} snimku"
    else:
        frames_str = "(pocet neznamy)"
    print(f"  {video_reader.width}x{video_reader.height} @ {video_reader.fps:.1f} FPS, {frames_str}")
    print(f"  Tracker: {tracker_name}")
    if kcf_info:
        print(
            f"  sigma={kcf_info.get('sigma')} lambda={kcf_info.get('lambda')} "
            f"lr={kcf_info.get('learning_rate')} padding={kcf_info.get('padding')}"
        )
    print("\n  CERVENA  = aktivni detekce")
    print("  ZLUTA    = slaba detekce")
    print("  ORANZOVA = KCF sleduje  (PSR ok)")
    print("  FIALOVA  = KCF ceka    (PSR pod prahem)\n")


def format_frame_info(
    frame_idx: int,
    tracks,
    cam_dx: float,
    cam_dy: float,
) -> str:
    """Sestav info radek nahore snimku."""
    n_det = sum(1 for t in tracks if t.state.value == "active")
    n_low = sum(1 for t in tracks if t.state.value == "low")
    n_kcf = sum(1 for t in tracks if t.state.value == "lost" and t.kcf_ok)
    n_wait = sum(1 for t in tracks if t.state.value == "lost" and not t.kcf_ok)
    return (
        f"Frame {frame_idx:05d} | "
        f"Det:{n_det} Low:{n_low} KCF:{n_kcf} Ceka:{n_wait}  "
        f"CMC dx:{cam_dx:+.1f} dy:{cam_dy:+.1f}"
    )


def print_final_stats(frame_idx: int, saved_count: int, elapsed_sec: float, output_dir: str) -> None:
    """Vytiskne zaverecne statistiky."""
    fps_avg = frame_idx / elapsed_sec if elapsed_sec > 0 else 0.0
    out_str = output_dir if output_dir else "(nic - nebyl zadan vystup)"
    print(f"\nHotovo! Zpracovano {frame_idx} snimku, ulozeno {saved_count} do {out_str}")
    print(f"Prumerne FPS: {fps_avg:.1f} (cas: {elapsed_sec:.1f} s)")
