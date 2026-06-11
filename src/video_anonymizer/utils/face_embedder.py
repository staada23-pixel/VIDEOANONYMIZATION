"""Face embedding pomoci YuNet landmarku (5 bodu: oci, nos, koutky ust).

Kazdy oblicej se prevede na vektor normalizovanych pomeru vzdalenosti mezi
landmarky, ktery je invariantni vuci scale a (castecne) vuci rotaci a pose.
Dva obliceje se povazuji za stejnou osobu, pokud je euklidovska vzdalenost
jejich vektoru pod thresholdem.
"""
from __future__ import annotations

import math


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def embed(landmarks: list[tuple[float, float]]) -> list[float]:
    """5 landmarku -> 10 normalizovanych pomeru (scale/rotacne invariantnich).

    Poradi landmarku: left_eye, right_eye, nose, left_mouth, right_mouth
    """
    if len(landmarks) < 5:
        return []
    le, re, no, lm, rm = landmarks[:5]

    # Vzdalenosti mezi vsemi 5 body (10 hodnot)
    raw = [
        _dist(le, re),
        _dist(le, no),
        _dist(le, lm),
        _dist(le, rm),
        _dist(re, no),
        _dist(re, lm),
        _dist(re, rm),
        _dist(no, lm),
        _dist(no, rm),
        _dist(lm, rm),
    ]
    # Normalizace interokularnim odstupem (LE-RE) — ten je u daneho cloveka
    # relativne konstantni bez ohledu na vzdalenost od kamery
    norm = raw[0]
    if norm < 1.0:
        return []
    return [v / norm for v in raw]


def distance(a: list[float], b: list[float]) -> float:
    """Euklidovska vzdalenost dvou embeddingu."""
    if not a or not b:
        return float("inf")
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def match(a: list[float], b: list[float], threshold: float = 0.35) -> bool:
    """True pokud oba embeddingy reprezentuji stejny oblicej."""
    return distance(a, b) < threshold
