# AGENTS.md — Video Anonymization

## Entry points

Two independent CLIs exist:

- **Recommended (package):** `python -m video_anonymizer.cli run --input ...` (requires `PYTHONPATH=src`)
- **Legacy (single-file):** `python video_anonymizer.py ...` or `run.cmd ...` (no PYTHONPATH needed, no YAML config, hardcoded defaults)

Subcommands: `run`, `info`, `blur-info`, `interactive` (alias `wizard`).

## Setup

```powershell
# Virtual env
bash setup_venv.sh            # creates .venv + installs deps
.venv\Scripts\activate

# PyYAML is NOT installed by setup_venv.sh — pip install separately if missing
pip install pyyaml

# CSRT/KCF trackers need opencv-contrib-python, NOT opencv-python
pip uninstall opencv-python; pip install opencv-contrib-python
```

## LPM SDK

Proprietary Eyedea SDK. Required at `externals/LPM/` with this layout:

```
externals/LPM/
├── LPM/lib/x64/lpm-v7.dll
├── modules-v7/x64/802-.../
├── modules-v7/config_camera_view_generic.ini
└── wrappers/python/lpm.py
```

Override: `$env:LPM_SDK_PATH = "C:\path\to\sdk"` (or `--lpm-sdk` on legacy CLI).

## Config hierarchy

`configs/config.yaml` is the main config. Per-tracker YAMLs at `configs/trackers/{csrt,kalman,kcf}.yaml` can be referenced from the main config. `--blur-config configs/blur.yaml` overrides the `blur:` section in the main config (merged recursively).

Config values are overridden by CLI flags when provided.

## Detection

| Backend | Requires | Notes |
|---------|----------|-------|
| `lpm` | LPM SDK + HASP license | Default |
| `mediapipe` | `pip install mediapipe` | Open-source fallback |

## Key CLI flags (package CLI)

| Flag | Effect |
|------|--------|
| `--input ""` or `--input-type webcam` | Webcam mode |
| `-b pixelate\|gaussian\|blackout\|none` | Blur method |
| `-t csrt\|kcf\|kalman` | Tracker |
| `-d lpm\|mediapipe` | Detector |
| `--output-video data/out.mp4` | Save MP4 |
| `--output-json data/out.json` | Per-frame JSON metadata |
| `--no-preview` | Disable live preview window |

## Notable quirks

- **No tests exist.** `tests/` directory is empty.
- `gaussian` blur `ksize` must be odd (auto-enforced to odd).
- `--blur-config` file merges into `blur:` section of main config (does NOT replace it).
- `tracker:` config section can reference per-type YAML via `config:` key (e.g. `config: configs/trackers/csrt.yaml`).
- Blur methods share `expand` (bbox expansion before anonymization, default 0.15).
- Visualisation colors are **BGR** (OpenCV convention), not RGB.

## GUI

Standalone launchers (no PYTHONPATH needed):
```powershell
python example_gui.py
python example_gui_advanced.py
python test_gui_simple.py   # no MediaPipe dependency
```

## Output reassembly

```powershell
ffmpeg -framerate 30 -i data/output_frames/frame_%06d.jpg -c:v libx264 out.mp4
```
