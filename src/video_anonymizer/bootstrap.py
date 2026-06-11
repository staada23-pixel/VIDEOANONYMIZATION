"""Nastaveni prostredi pred spustenim CLI (PYTHONPATH, LPM SDK, working dir)."""
from __future__ import annotations

import os
import sys


def project_root() -> str:
    """.../project/ odvozene od umisteni balicku."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def find_lpm_wrappers(root: str | None = None) -> str | None:
    """Najdi adresar s er.py a lpm.py."""
    root = root or project_root()
    candidates: list[str] = []

    parent = os.path.dirname(root)
    if os.path.isdir(parent):
        for entry in os.listdir(parent):
            if entry.upper().startswith("LPM-"):
                candidates.append(os.path.join(parent, entry, "wrappers", "python"))

    candidates.extend([
        os.path.join(root, "externals", "LPM", "wrappers", "python"),
        os.path.join(
            r"C:\Users\face\Desktop\Praxe 2026\EYEDEA PROJECT",
            "LPM-v7.9.1-2026-04-08-Windows-10-x64-hasp10.2",
            "wrappers", "python",
        ),
    ])

    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "lpm.py")):
            return os.path.abspath(cand)
    return None


def setup_environment(root: str | None = None) -> str:
    """
    Pridej src + LPM do sys.path a prepni CWD na project root.
    Vrati absolutni cestu k project root.
    """
    root = os.path.abspath(root or project_root())
    src = os.path.join(root, "src")

    if src not in sys.path:
        sys.path.insert(0, src)

    lpm = find_lpm_wrappers(root)
    if lpm and lpm not in sys.path:
        sys.path.insert(0, lpm)

    try:
        os.chdir(root)
    except OSError:
        pass

    return root
