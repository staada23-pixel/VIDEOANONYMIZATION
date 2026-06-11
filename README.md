# Video Anonymizer

Nastroj pro **detekci obliceju + tracking + anonymizaci** ve videu.

- **Detektor**: Eyedea LPM (s HASP) / OpenCV DNN YuNet (bez HASP, obličeje) / HOG (postavy) / Haar cascade
- **Tracker**: KCF (s template-matching fallbackem proti zamrznuti)
- **Anonymizace**: mosaic, blur, black box, solid, none
- **Bez HASP**: plně funkční s `YuNet` detektorem

## Požadavky
- Python 3.10+
- Windows 10+ / Linux
- Volitelně: Eyedea LPM SDK v7.9+ s HASP dongle (pro špičkovou kvalitu)

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

### Interaktivní menu (doporučeno pro první pokus)
```bat
.\run.ps1
```
nebo
```bat
python -m video_anonymizer --interactive
```

### Příkazová řádka
```bat
:: TUI menu (bez argumentu)
.\run.ps1

:: Bodycam s YuNet (OBLIČEJE, bez HASP)
.\run.ps1 --input video.mp4 --detector face --anon-method mosaic

:: Uložit anonymizované MP4
.\run.ps1 --input video.mp4 --detector face --out-video out.mp4 --no-display

:: Webcam v realnem case
.\run.ps1 --input 0 --detector face --anon-method blur

:: LPM (vyžaduje HASP dongle)
.\run.ps1 --input video.mp4 --detector lpm
```

## Parametry CLI

| Přepínač | Výchozí | Popis |
|----------|---------|-------|
| `--input` | - | Cesta k videu / `0` pro webcam (bez něj → TUI) |
| `--detector` | `auto` | `auto`/`lpm`/`face`/`hog`/`haar` |
| `--anon-method` | `mosaic` | `none`/`mosaic`/`blur`/`black`/`solid` |
| `--anon-strength` | `15` | Síla efektu (1-50) |
| `--out-video` | - | Cesta pro výstupní anonymizované MP4 |
| `--out-frames` | - | Složka pro jednotlivé JPG |
| `--no-display` | `false` | Bez okna (jen uložit) |
| `--no-boxes` | `false` | Bez vykreslování boxů (čistý výstup) |
| `--interactive` | `false` | Vynutí TUI menu |
| `--config` | `configs/config.yaml` | Cesta ke konfiguraci |

## Detektory — co použít kdy

| Detektor | Co detekuje | HASP | Kvalita | Rychlost | Doporučení |
|----------|-------------|------|---------|----------|------------|
| `lpm` | Osoby + atributy | ✅ Ano | ★★★★★ | ~30 FPS | Produkční nasazení |
| `face` | **Obličeje** | ❌ Ne | ★★★★ | ~25 FPS | **Bodycam, demo** |
| `hog` | Celé postavy | ❌ Ne | ★★ | ~15 FPS | Crowd counting |
| `haar` | Obličeje | ❌ Ne | ★★★ | ~40 FPS | Fallback |

> **Doporučení pro bodycam**: `--detector face` (YuNet DNN). Detekuje obličeje (ne celé postavy), nevyžaduje HASP, běží 25+ FPS.

## Struktura projektu
```
project/
├── configs/
│   ├── config.yaml
│   └── trackers/{kcf,visualisation}.yaml
├── src/video_anonymizer/
│   ├── cli.py                # entry point
│   ├── __main__.py
│   ├── tui.py                # interaktivní menu
│   ├── detection/
│   │   ├── lpm_wrapper.py    # LPM SDK (lazy import)
│   │   ├── face_detector.py  # YuNet + Haar
│   │   ├── hog_detector.py
│   │   └── structures.py
│   ├── tracking/
│   │   ├── kcf.py            # KCF + template matching fallback
│   │   ├── byte_tracker.py
│   │   └── base_tracker.py
│   ├── io/
│   │   ├── video_reader.py
│   │   └── frame_writer.py
│   └── utils/
│       ├── anonymizer.py     # mosaic, blur, black, solid
│       ├── cmc.py
│       └── ...
├── tests/
│   ├── test_anonymizer.py
│   ├── test_pipeline_mock.py
│   ├── test_hog_real.py
│   ├── test_bodycam_full.py
│   ├── bench_face_params.py
│   └── make_demo_video.py
├── models/                   # YuNet ONNX (auto-download)
├── run.ps1 / run.bat
└── setup_venv.bat / .sh
```

## Testy
```bat
:: Unit testy anonymizace
python tests\test_anonymizer.py

:: Mock pipeline test
python tests\test_pipeline_mock.py

:: Benchmark detektoru
python tests\bench_face_params.py

:: Plny smoke test na bodycam videu
python tests\test_bodycam_full.py
```

## Výstup testu na bodycam videu (1152×632 @ 25 fps, 599 snímků)
```
Detektor: FaceDetector (YuNet)
Anonymizer: mosaic (sila=18)
  detekci celkem: 653 (1.09/snímek)
  tracku celkem:  1015 (1.69/snímek)
  processing:     8.0 FPS
  video out:      project/output/bodycam_anonymized.mp4
```

Všechny výstupy (videa, snímky) se standardně ukládají do `project/output/` (gitignored).

Výsledek: obličeje pixelovány, identita osob chráněna, kamera overlay a kontext scény zachovány.
