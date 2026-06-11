"""Test TUI menu end-to-end s mockovanym vstupem."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch
from video_anonymizer import tui


def test_tui_default_detector_is_face():
    """Default detektor v TUI by mel byt face DNN."""
    # inputs:
    #   1) video: 2 = demo.mp4
    #   2) detector: Enter = default (face)
    #   3) tracker: Enter = default (KCF)
    #   4) anon: Enter = default (mosaic)
    #   5) strength: 20
    #   6) save video: y
    #   7) save frames: n
    #   8) show window: n
    #   9) backfill: n
    #  10) show boxes: n
    #  11) spustit: y
    inputs = ["2", "", "", "", "20", "y", "n", "n", "n", "n", "y"]
    with patch("builtins.input", side_effect=inputs):
        opts = tui.run_interactive()

    assert opts["detector"] == "face", f"Ocekavano 'face', dostal {opts['detector']!r}"
    assert opts["anon_method"] == "mosaic", f"Ocekavano 'mosaic', dostal {opts['anon_method']!r}"
    assert opts["anon_strength"] == 20
    assert opts["save_video"] is True
    assert opts["out_video"].endswith("anonymized.mp4")
    assert os.path.isabs(opts["config"]), f"Config neni absolutni: {opts['config']}"
    assert os.path.isfile(opts["config"]), f"Config neexistuje: {opts['config']}"
    assert os.path.isabs(opts["out_video"]), f"Out video neni absolutni: {opts['out_video']}"
    print(f"  ok  default detektor=face, config={opts['config']}")


def test_tui_lpm_detector():
    """TUI umi zvolit LPM detektor."""
    # inputs:
    #   1) video: 4 = bodycam
    #   2) detector: 2 = LPM
    #   3) tracker: Enter = default (KCF)
    #   4) anon: Enter = default (mosaic)
    #   5) strength: Enter = default (15)
    #   6) save video: n
    #   7) save frames: n
    #   8) show window: n
    #   9) backfill: n
    #  10) show boxes: n
    #  11) spustit: y
    inputs = ["4", "2", "", "", "", "n", "n", "n", "n", "n", "y"]
    with patch("builtins.input", side_effect=inputs):
        opts = tui.run_interactive()

    assert opts["detector"] == "lpm"
    assert opts["save_video"] is False
    print(f"  ok  LPM detektor, save_video=False")


def test_tui_q_exits():
    """TUI umi 'q' pro zruseni."""
    with patch("builtins.input", side_effect=["q"]):
        try:
            tui.run_interactive()
        except SystemExit as e:
            assert e.code == 0
            print("  ok  'q' na video vstup -> SystemExit(0)")
            return
    raise AssertionError("SystemExit nebyl vyvolan")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            raise
    print(f"\n{len(tests)} TUI testu proslo.")
