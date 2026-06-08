# Video Anonymization

Anonymizace obličejů ve videu. Pipeline: **detekce (LPM SDK / MediaPipe) → tracker (CSRT / KCF / Kalman) → anonymizace (pixelace / gaussian blur / blackout) → zápis snímků + videa + JSON metadat**.

Dva nezávislé CLIs:
- **Package CLI** (`python -m video_anonymizer.cli run`) — YAML konfigurace, MediaPipe i LPM, JSON metadata, 4 subcommands
- **Legacy CLI** (`python video_anonymizer.py`) — single-file, bez YAML, jen LPM, hardcoded defaulty

## Struktura

```
VIDEOANONYMIZATION/
├── src/video_anonymizer/
│   ├── cli.py                     # Hlavní CLI (argparse, 4 subcommands, pipeline loop)
│   ├── __init__.py
│   ├── io/
│   │   ├── video_reader.py        # Samonosné čtení (video / obrázek / adresář / webcam)
│   │   └── frame_writer.py        # FrameWriter, VideoWriter, JSONWriter, MultiWriter
│   ├── detection/
│   │   ├── detection_model.py     # Detection dataclass
│   │   ├── lpm_wrapper.py         # LPM SDK wrapper (cffi)
│   │   └── mediapipe_detector.py  # MediaPipe detektor (face + hands)
│   ├── tracking/
│   │   ├── track.py               # Track dataclass
│   │   ├── csrt.py                # CSRT tracker (OpenCV contrib)
│   │   ├── kcf.py                 # KCF tracker
│   │   ├── kalman.py              # Kalman‑Bucy tracker (8‑state)
│   │   └── iou_tracker.py         # IoU multi‑object tracker (progr. použití)
│   ├── anonymizer/
│   │   └── blur.py                # pixelate / gaussian / blackout / expand_bbox
│   ├── utils/
│   │   ├── logging.py             # Centrální logging setup
│   │   └── overlap_fn.py          # IoU, NMS, is_likely_face
│   └── gui/
│       └── detection_viewer.py    # PyQt6 GUI viewer (bbox vizualizace)
├── configs/
│   ├── config.yaml                # Hlavní konfigurace (LPM, detekce, tracker, blur, output)
│   ├── blur.yaml                  # Per‑method blur parametry (merge do config.yaml)
│   └── trackers/
│       ├── kalman.yaml            # Kalman‑specific (process/measurement noise)
│       ├── kcf.yaml               # Minimal config
│       └── visualisation.yaml     # Barvy (BGR) + label formát pro bbox overlay
├── externals/
│   └── LPM/                       # LPM SDK (vložit distribuci — aktuálně prázdné)
├── data/
│   ├── input/                     # Vstupní videa / obrázky
│   ├── output_frames/             # Výstupní snímky (gitignored)
│   └── output/                    # Alternativní výstup (gitignored)
├── video_anonymizer.py            # Legacy single‑file CLI (nevyžaduje PYTHONPATH)
├── run.cmd                        # Windows launcher pro legacy CLI
├── example_gui.py                 # Minimální GUI launcher
├── example_gui_advanced.py        # GUI + synthetic test image
├── test_gui_simple.py             # GUI test bez MediaPipe
├── tests/                         # prázdné (rezerva)
├── requirements.txt
├── setup_venv.sh
└── AGENTS.md
```

## Instalace

```bash
bash setup_venv.sh            # vytvoří .venv a nainstaluje závislosti
source .venv/Scripts/activate  # Windows
source .venv/bin/activate      # Linux/macOS
```

PyYAML se v `setup_venv.sh` neinstaluje — `pip install pyyaml` zvlášť, pokud ho setup nepřidá.

`setup_venv.sh` instaluje navíc `filterpy` (není v `requirements.txt`).

### Závislosti

- `opencv-contrib-python>=4.8.0` (CSRT/KCF vyžadují contrib, ne plain `opencv-python`)
- `PyQt6>=6.6.0` (jen pro GUI)
- `mediapipe>=0.10.0` (nepovinné — open‑source fallback detektor)
- `pyyaml>=6.0`
- `numpy`, `pillow`, `cffi`

## LPM SDK setup

`externals/LPM/` musí obsahovat LPM 7.9.1 distribuci v tomto layoutu:

```
externals/LPM/
├── LPM/lib/x64/lpm-v7.dll
├── modules-v7/x64/802-.../
├── modules-v7/config_camera_view_generic.ini
└── wrappers/python/lpm.py
└── wrappers/python/er.py
```

**Aktuálně je `externals/LPM/` prázdné.** SDK je nutné do něj nakopírovat, nebo nastavit `LPM_SDK_PATH=/cesta/k/sdk`. Pokud SDK není k dispozici, `LPMDetector` je inertní (loguje warning, vrací prázdné detekce) a pipeline běží dál — doporučuje se použít `--detector mediapipe`.

## Použití — Package CLI (doporučeno)

Vyžaduje `PYTHONPATH=src`:

```bash
export PYTHONPATH=src      # Linux/macOS
$env:PYTHONPATH = "src"    # PowerShell
```

### Subcommands

| Subcommand | Popis |
|------------|-------|
| `run` | Hlavní anonymizační pipeline |
| `info` | Vypíše aktuální konfiguraci |
| `blur-info` | Vypíše dostupné blur metody |
| `interactive` / `i` / `wizard` | 13‑krokový interaktivní průvodce |

### Příklady `run`

```bash
# video → anonymizované video + snímky + JSON metadata
python -m video_anonymizer.cli run \
    -i video.mp4 \
    --output-video data/out.mp4 \
    --output-frames data/out_frames/ \
    --output-json data/out.json \
    -t csrt -d lpm -b pixelate --no-preview

# jeden obrázek → anonymizovaný obrázek + json
python -m video_anonymizer.cli run \
    -i shot.jpg --output-frames data/out_img/ --output-json data/out_img.json

# adresář snímků → anonymizované snímky + mp4
python -m video_anonymizer.cli run \
    -i data/raw/ --output-frames data/out_frames/ --output-video data/out.mp4 \
    -b blackout

# webcam (prázdný --input)
python -m video_anonymizer.cli run --input-type webcam

# interaktivní průvodce
python -m video_anonymizer.cli interactive

# info o konfiguraci
python -m video_anonymizer.cli info
python -m video_anonymizer.cli blur-info
```

### Argumenty `run`

| Flag | Význam | Default |
|------|--------|---------|
| `--input / -i` | Video / obrázek / adresář / (prázdné=webcam) | — |
| `--input-type` | `auto` / `video` / `image` / `image_dir` / `webcam` | `auto` |
| `--output-frames` | Adresář pro JPEG sekvenci (`frame_%06d.jpg`) | z `config.yaml` |
| `--output-video` | Výstupní video (.mp4/.avi) | z `config.yaml` (null=skip) |
| `--output-json` | Per‑frame JSON metadata (detekce + track + anonymized bbox) | z `config.yaml` (null=skip) |
| `--fps` | FPS výstupního videa | z video meta |
| `--tracker / -t` | `csrt` / `kcf` / `kalman` | z `config.yaml` |
| `--detector / -d` | `lpm` / `mediapipe` | z `config.yaml` |
| `--redetect-every` | Re‑detekce každých N framů | z `config.yaml` |
| `--blur-method / -b` | `pixelate` / `gaussian` / `blackout` / `none` | z `config.yaml` |
| `--blur-config` | YAML s per‑method blur parametry (merge do `blur:` sekce) | `configs/blur.yaml` |
| `--no-anonymize` | Vypne anonymizaci (jen bbox overlay) | off |
| `--no-preview` | Vypne live preview okno | off |
| `--no-boxes` | Vypne bbox overlay | off |
| `--start-frame` | První frame (0‑indexed) | `0` |
| `--end-frame` | Poslední frame (inclusive) | null (=do konce) |
| `--config / -c` | Hlavní YAML config | `configs/config.yaml` |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |

## Použití — Legacy CLI (single‑file)

Nevyžaduje `PYTHONPATH`, nepodporuje YAML config ani MediaPipe — pouze LPM detektor.

```bash
# Windows launcher (aktivuje .venv automaticky)
run.cmd video.mp4
run.cmd video.mp4 --tracker kcf --blur-method gaussian
run.cmd --wizard                           # interaktivní průvodce

# Přímé spuštění
python video_anonymizer.py video.mp4
python video_anonymizer.py video.mp4 --tracker kcf --blur-method gaussian
python video_anonymizer.py                 # automaticky spustí wizard
```

### Legacy‑specific flags

| Flag | Význam | Default |
|------|--------|---------|
| `--wizard` | Vynutí interaktivní průvodce | off |
| `--lpm-sdk` | Cesta k LPM SDK | env / `externals/LPM/` / `../LPM/` |
| `--lost-frames` | Framů před re‑akvizicí | `15` |
| `--search-radius` | Max pixel distance pro re‑akvizici | `200` |
| `--module-id` | LPM module ID | `802` |
| `--min-confidence` | Min confidence detekce | `0.40` |
| `--initial-scan` | Scan‑ahead framů při startu | `30` |
| `--pixel-block` | Velikost dlaždice pro pixelate | `18` |
| `--gaussian-ksize` | Kernel size (auto‑enforced lichý) | `21` |
| `--bbox-expand` | Expanze bboxu před anonymizací | `0.15` |
| `--no-blur` | Vypne anonymizaci | off |
| `--draw-bbox` | Vykreslí bbox overlay | off |
| `--no-preview` | Vypne live preview | off |
| `--save-video` | Cesta k výstupnímu MP4 | null (skip) |

Výstupní snímky: `output_video_frames/frame_%05d.jpg` (5 číslic, package CLI používá 6).

## Konfigurační hierarchie

1. `configs/config.yaml` — hlavní konfigurace (všechny sekce)
2. `--blur-config configs/blur.yaml` — merge do `blur:` sekce (rekurzivní merge, **nenahrazuje** celou sekci)
3. Per‑tracker YAML v `configs/trackers/` — lze referencovat z `tracker.config` v hlavním configu
4. CLI flagy přepisují cokoli z configu

### configs/config.yaml

```yaml
lpm:
  module_id: 802
  version: 7
  det_compute_on_gpu: false
  det_num_threads: 1
detection:
  backend: lpm              # lpm | mediapipe
  min_confidence: 0.4
  redetect_every_n: 15
  min_face_ratio: 0.03
  nms_iou_threshold: 0.3
tracker:
  type: csrt                # csrt | kcf | kalman
  lost_timeout: 20
  config: configs/trackers/csrt.yaml
blur:
  method: pixelate          # pixelate | gaussian | blackout | none
  block: 18
  ksize: 51                 # musí být liché
  color: [0, 0, 0]          # BGR
  expand: 0.15
visualisation:
  enabled: true
  show_bbox: true
  show_label: true
  bbox_color: [0, 255, 0]   # BGR — zelená
  bbox_thickness: 2
  font_scale: 0.5
  lost_bbox_color: [0, 0, 255]  # BGR — červená
input:
  type: auto
  start_frame: 0
  end_frame: null
output:
  frames: data/output_frames
  video: null               # null = nezapisovat
  json: null                # null = nezapisovat
  jpg_quality: 90
  video_codec: mp4v
```

### configs/trackers/visualisation.yaml — label formát

```yaml
colors:
  active: [0, 255, 0]    # zelená — detekovaný
  lost: [0, 0, 255]      # červená — tracker ztratil
  text: [255, 255, 255]  # bílá — text labelu
label:
  show_id: true
  show_confidence: true
  show_lost_count: true
  font_scale: 0.6
```

**Barvy jsou v BGR** (OpenCV konvence), ne RGB.

## Anonymizační metody

| Metoda | Parametry | Popis |
|--------|-----------|-------|
| `pixelate` | `block: 18` | Mozaika — dlaždice o `block` px |
| `gaussian` | `ksize: 51` | Gaussovský blur přes ROI (`ksize` musí být liché) |
| `blackout` | `color: [0,0,0]` (BGR) | Plný barevný obdélník |
| `none` | — | Bez anonymizace (jen bbox overlay) |

Všechny metody sdílejí `expand: 0.15` (rozšíření bboxu před anonymizací).

**Poznámka:** V package CLI (`blur.py`) není `gaussian` ksize auto‑enforced na lichý — je nutné zadat liché číslo ručně. V legacy CLI (`video_anonymizer.py`) je auto‑enforced pomocí `k = ksize | 1`.

## JSON výstup

Při použití `--output-json` se generuje per‑frame JSON v tomto formátu:

```json
{
  "meta": {
    "input": "video.mp4",
    "input_type": "video",
    "tracker": "csrt",
    "detector": "lpm",
    "blur_method": "pixelate"
  },
  "frames": [
    {
      "frame": 0,
      "detections": [
        {"x": 100, "y": 50, "w": 80, "h": 100, "confidence": 0.95, "label": "Face"}
      ],
      "track": {
        "id": 1,
        "bbox": [100, 50, 180, 150],
        "confidence": 0.95,
        "lost_frames": 0,
        "active": true
      },
      "anonymized_bbox": [85, 40, 195, 160]
    }
  ]
}
```

## GUI viewer

```bash
python example_gui.py
python example_gui_advanced.py        # synthetic test image
python test_gui_simple.py             # bez MediaPipe
```

GUI je postaveno na PyQt6 a podporuje:
- Načtení obrázku nebo videa (file dialog / drag‑and‑drop)
- Detekci obličejů a rukou (MediaPipe)
- Vykreslení bboxů s labely
- Slideshow pro videa se sliderem
- Tabulku detekcí

## IoU multi‑object tracker

`src/video_anonymizer/tracking/iou_tracker.py` je jednoduchý IoU‑based multi‑object tracker pro programátorské použití (není zapojen do CLI pipeline). Přiřazuje detekce k existujícím trackům podle IoU překryvu, nové tracky vznikají z nepřiřazených detekcí, ztracené se ruší po `max_lost` framech.

## Reassemble do MP4

```bash
# Package CLI — 6‑místné číslování (frame_%06d.jpg)
ffmpeg -framerate 30 -i data/output_frames/frame_%06d.jpg -c:v libx264 out.mp4

# Legacy CLI — 5‑místné číslování (frame_%05d.jpg)
ffmpeg -framerate 30 -i output_video_frames/frame_%05d.jpg -c:v libx264 out.mp4
```

## OpenCV poznámka

`cv2.TrackerCSRT_create` a `cv2.TrackerKCF_create` vyžadují `opencv-contrib-python`, ne obyčejný `opencv-python`:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## Verzování

| Komponenta | Verze |
|------------|-------|
| LPM SDK | 7.9.1 |
| Python | 3.14 (testováno), 3.10+ by mělo fungovat |
| OpenCV | 4.8+ (contrib) |
