"""Test anonymizacnich metod.

Overuje ze Anonymizer:
  - maze mimo obraz (clipping)
  - vraci frame spravneho typu/rozmeru
  - vsechny metody produkuji vystup
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from video_anonymizer.utils.anonymizer import Anonymizer, METHODS


def test_methods_registered():
    assert "none" in METHODS
    assert "mosaic" in METHODS
    assert "blur" in METHODS
    assert "black" in METHODS
    assert "solid" in METHODS


def test_none_does_nothing():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    a = Anonymizer("none")
    out = a.apply(img.copy(), (10, 10, 50, 50))
    assert np.array_equal(out, img)


def test_black_makes_black():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    a = Anonymizer("black")
    out = a.apply(img.copy(), (10, 10, 50, 50))
    roi = out[10:50, 10:50]
    assert np.all(roi == 0), f"black method failed: max={roi.max()}"


def test_solid_makes_color():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    a = Anonymizer("solid", color=(255, 0, 0))
    out = a.apply(img.copy(), (10, 10, 50, 50))
    roi = out[10:50, 10:50]
    assert np.all(roi[..., 0] == 255) and np.all(roi[..., 1] == 0) and np.all(roi[..., 2] == 0)


def test_mosaic_blocks():
    img = np.random.RandomState(0).randint(0, 256, (200, 200, 3), dtype=np.uint8)
    a = Anonymizer("mosaic", strength=20)
    out = a.apply(img.copy(), (40, 40, 160, 160))
    # ROI=120x120, interni block=20, scale=6 → 6×6 vystupnich bloku po 6×6 pixelech
    roi = out[40:160, 40:160]
    # Dva body ve stejnem 6×6 bloku (oba v [0..5, 0..5])
    p1 = roi[0, 0]
    p2 = roi[3, 4]
    np.testing.assert_array_equal(p1, p2,
        err_msg=f"body v jednom bloku se maji rovnat: {p1} vs {p2}")
    # A naopak: dva body v ruznych blocich by se mely lisit
    p3 = roi[0, 0]
    p4 = roi[6, 6]
    assert not np.array_equal(p3, p4), \
        f"ruzne bloky by se mely lisit, ale oba daly {p3}"


def test_blur_smooths():
    img = np.random.RandomState(0).randint(0, 256, (200, 200, 3), dtype=np.uint8)
    a = Anonymizer("blur", strength=10)
    out = a.apply(img.copy(), (40, 40, 160, 160))
    # Variance v ROI by mela byt vyssi v orig
    orig_var = float(np.var(img[40:160, 40:160]))
    blur_var = float(np.var(out[40:160, 40:160]))
    assert blur_var < orig_var, f"blur variance {blur_var} not < orig {orig_var}"


def test_clipping_outside_image():
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    a = Anonymizer("black")
    # Box castecne mimo obraz
    out = a.apply(img.copy(), (-20, -20, 50, 50))
    # nemelo by spadnout, vystup musi byt 100x100
    assert out.shape == (100, 100, 3)


def test_invalid_method_raises():
    try:
        Anonymizer("nope")
    except ValueError:
        return
    assert False, "should raise on invalid method"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            raise
    print(f"\n{len(tests)} testů prošlo.")
