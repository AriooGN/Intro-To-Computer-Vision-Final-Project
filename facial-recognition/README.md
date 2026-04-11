# Real-Time Facial Recognition & Detection

Python application that runs **real-time face detection** with OpenCV’s DNN SSD model and **identity recognition** with **DeepFace** against a local gallery under `known_faces/`.

## Setup

```bash
cd facial-recognition
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On first run, OpenCV DNN weights are downloaded into `models/`. DeepFace will download model weights on demand.

## Known faces layout

```
known_faces/
  person_name/
    img1.jpg
    img2.jpg
```

Use **1–5 reference images per person** for best results. Supported image formats follow OpenCV/DeepFace (e.g. `.jpg`, `.png`).

## Run

```bash
python main.py
```

### CLI options

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `0` | Webcam index |
| `--model` | `VGG-Face` | `VGG-Face`, `Facenet`, or `ArcFace` |
| `--det-conf` | `0.5` | Minimum detection confidence |
| `--recognition-interval` | `5` | Run recognition every *N* frames |
| `--min-face` | `40` | Ignore faces smaller than this (px) at full resolution |
| `--detection-scale` | `0.5` | Resize factor for detection (faster, then map boxes back) |

### Keyboard

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Cycle model: VGG-Face → Facenet → ArcFace |
| `s` | Save screenshot (project root) |
| `a` | Register **largest** face: prompts for name in the terminal, saves crop under `known_faces/<name>/`, clears DeepFace’s `representations*.pkl` cache |

## Configuration

Edit `config.py` for default paths, per-model **recognition distance thresholds** (aligned with DeepFace verification), and `DEEPFACE_ALIGN` (cropped faces default to `False`).

## Architecture

- **`main.py`** — `VideoCapture` loop, frame skipping for recognition, HUD, keyboard UI.
- **`detector.py`** — OpenCV DNN face detector + confidence; optional auto-download of prototxt/caffemodel.
- **`recognizer.py`** — Background thread; **`DeepFace.find()`** on each cropped face; clears gallery pickle cache when identities change.
- **`utils.py`** — FPS, scaling boxes to full resolution, crop/draw helpers.
- **`config.py`** — Thresholds and constants.

Detection runs on **every frame** (optionally downscaled). Recognition runs **every N frames** in a **worker thread** so the preview stays responsive. Labels are held until the next consistent multi-face result so the UI does not flicker when async results lag by one frame.

## Notes

- **GPU**: TensorFlow/DeepFace will use GPU if configured in your environment; otherwise CPU.
- **Empty gallery**: All faces show as **Unknown** until you add references under `known_faces/`.
- **Performance**: Lower `--detection-scale`, increase `--recognition-interval`, or use a lighter DeepFace model to improve FPS.
