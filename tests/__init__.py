"""Testy a testovací data projektu Video Anonymizer.

Testovací soubory:
- test_anonymizer.py — unit testy anonymizačních metod
- test_pipeline_mock.py — end-to-end test pipeline s mock detekcemi
- test_bodycam_full.py — smoke test na reálném bodycam videu
- test_hog_real.py — test HOG detektoru na syntetické postavě
- test_tui.py — test interaktivního menu s mock vstupem
- bench_face_params.py — benchmark parametrů face detektoru
- make_demo_video.py — generátor syntetického dema
- test_helpers.py — pomocné funkce pro testy

Testovací videa:
- demo.mp4 — syntetické 5s demo video (vygenerováno make_demo_video.py)
- *.mp4 od uživatele (WhatsApp, bodycam, ...) — reálná testovací data
"""
