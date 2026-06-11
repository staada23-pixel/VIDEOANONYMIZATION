"""Video Anonymizer — detekce, tracking a anonymizace obličejů ve videu.

Hlavní balíček projektu. Struktura:
  cli.py         — CLI entry point a hlavní pipeline
  __main__.py    — `python -m video_anonymizer` entry
  tui.py         — interaktivní textové menu
  bootstrap.py   — inicializace prostředí (PYTHONPATH, LPM SDK)
  video_catalog.py — vyhledávání videí v projektu
  detection/     — detektory (YuNet, Haar, HOG, LPM)
  tracking/      — trackery (KCF, ByteTracker)
  io/            — čtení/ukládání videí a snímků
  utils/         — anonymizace, CMC, overlap, logging
"""
