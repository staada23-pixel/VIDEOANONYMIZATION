# Video Anonymizer

Nástroj pro **detekci obličejů + tracking + anonymizaci** ve videu.
Podpora bodycam záznamů, webcam v reálném čase i offline zpracování.

- **Detektor**: YuNet DNN (bez HASP, obličeje) / Eyedea LPM (s HASP) / HOG (postavy) / Haar cascade
- **Tracker**: ByteTracker (3-kola IoU matching) + KCF (template-matching fallback)
- **Anonymizace**: mosaic, blur, black box, solid, none
- **Backfill**: zpětné doplnění chybějících detekcí lineární interpolací

## Požadavky

- Python 3.10+
- Windows 10+ / Linux
- Volitelně: Eyedea LPM SDK v7.9+ s HASP dongle

## Instalace

### Windows
```bat
setup_venv.bat
```

### Linux/Mac
```bash
chmod +x setup_venv.sh && ./setup_venv.sh
```

## Rychlé spuštění

### Interaktivní menu (doporučeno)
```bat
python -m video_anonymizer
```

### Příkazová řádka

```bat
:: Bodycam — YuNet + mozaika + backfill
python -m video_anonymizer --input video.mp4 --detector face --backfill --out-video vystup.mp4 --no-display

:: Rychlý test (jen 300 snímků)
python -m video_anonymizer --input video.mp4 --detector face --max-frames 300 --no-display

:: Webcam v reálném čase
python -m video_anonymizer --input 0 --detector face --anon-method blur

:: LPM (vyžaduje HASP dongle)
python -m video_anonymizer --input video.mp4 --detector lpm

:: Jen bounding boxy (žádná anonymizace) — test detekce
python -m video_anonymizer --input video.mp4 --detector face --anon-method none
```

## Parametry CLI

| Přepínač | Výchozí | Popis |
|----------|---------|-------|
| `--input` | — | Cesta k videu / `0` pro webcam (bez něj → TUI) |
| `--detector` | `auto` | `auto`/`lpm`/`face`/`hog`/`haar` |
| `--tracker` | `KCF` | `KCF`/`CSRT`/`MIL`/`VIT` |
| `--anon-method` | `mosaic` | `none`/`mosaic`/`blur`/`black`/`solid` |
| `--anon-strength` | `15` | Síla efektu (1–50) |
| `--out-video` | — | Cesta pro výstupní anonymizované MP4 |
| `--out-frames` | — | Složka pro jednotlivé JPG |
| `--no-display` | — | Bez okna (jen uložit) |
| `--no-boxes` | — | Bez vykreslování boxů (čistý anonymizovaný obraz) |
| `--backfill` | — | Po forward passu doplní chybějící detekce zpětně |
| `--max-frames` | `0` | Zpracovat max N snímků (0 = celé video) |
| `--interactive` | — | Vynutí TUI menu |
| `--dry-run` | — | Rychlý test na demo videu (50 snímků) |
| `--config` | `configs/config.yaml` | Cesta ke konfiguraci |

## Detektory — co použít kdy

| Detektor | Co detekuje | HASP | Kvalita | Rychlost | Doporučení |
|----------|-------------|------|---------|----------|------------|
| `face` | **Obličeje** | ❌ Ne | ★★★★ | ~25 FPS | **Bodycam, demo, nejčastější** |
| `lpm` | Osoby + atributy | ✅ Ano | ★★★★★ | ~30 FPS | Produkční nasazení |
| `hog` | Celé postavy | ❌ Ne | ★★ | ~15 FPS | Crowd counting |
| `haar` | Obličeje | ❌ Ne | ★★★ | ~40 FPS | Fallback |

> **Doporučení**: `--detector face --backfill` — YuNet DNN bez HASP s dohledáním chybějících detekcí.

## Backfill

Backfill je klíčová funkce pro bodycam záběry. Po přečtení celého videa
dopředu se vrátí a lineární interpolací mezi známými detekcemi vyplní
mezery, kde tracking nezvládl udržet obličej.

- Funguje jen pro tracky s **≥ 3 detekcemi** (KCF drift se nebackfilluje)
- Maximální **30 snímková mezera** mezi detekcemi
- **Žádná CMC** — čistě lineární interpolace mezi boxy

## Testy

```bat
:: Unit testy anonymizace
python tests\test_anonymizer.py

:: Mock pipeline test
python tests\test_pipeline_mock.py

:: Benchmark detektoru
python tests\bench_face_params.py

:: Test na bodycam videu
python tests\test_bodycam_full.py
```

## Struktura

```
project/
├── GUIDE.txt                    Tento průvodce
├── cli.py                       Hlavní entry point
├── pyproject.toml               Závislosti
├── setup_venv.bat / .sh         Instalace
├── run.bat / run.ps1            Rychlé spuštění
├── src/video_anonymizer/
│   ├── detection/               Detektory
│   │   ├── face_detector.py     YuNet + Haar
│   │   ├── hog_detector.py
│   │   ├── lpm_wrapper.py       LPM SDK
│   │   └── structures.py
│   ├── tracking/
│   │   ├── byte_tracker.py      ByteTracker (3-kola IoU)
│   │   ├── kcf.py               KCF + template-matching fallback
│   │   └── base_tracker.py
│   ├── io/
│   │   ├── video_reader.py
│   │   └── frame_writer.py
│   └── utils/
│       ├── anonymizer.py
│       ├── backfill.py          Lineární backfill
│       ├── cmc.py               Camera motion compensation
│       ├── face_embedder.py
│       └── overlap.py
├── configs/                     YAML konfigurace
├── models/                      ONNX modely (auto-download)
└── tests/
```

## Poznámky

- Výstupy se ukládají do `output/` (gitignored)
- YuNet model se stáhne automaticky při prvním spuštění
- Detailní průvodce: `GUIDE.txt`
