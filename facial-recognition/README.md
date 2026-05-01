# Real-Time Face Detection & Recognition

Python app that combines **face detection** (find faces in the camera feed) with **face recognition** (guess who each face is using a local gallery).

## Detection vs recognition

These are separate steps in a typical pipeline:

| | **Detection** | **Recognition** |
|---|---|---|
| **Question** | *Where* are faces in the image? | *Who* is this face (if anyone we know)? |
| **Output** | Bounding boxes + confidence scores | Name / **Unknown**, based on similarity to `known_faces/` |
| **This project** | OpenCV DNN SSD (`detector.py`) | DeepFace embeddings + gallery search (`recognizer.py`) |

Detection does **not** need enrolled identities—it only finds candidate face regions. Recognition runs **on cropped faces** from detection and compares them to your stored references; without a gallery, every detected face is labeled **Unknown**.

## Setup

```bash
cd facial-recognition
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On first run, OpenCV DNN weights are downloaded into `models/`. DeepFace downloads model weights on demand.

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
| `--det-conf` | `0.5` | Minimum **detection** confidence |
| `--recognition-interval` | `5` | Run **recognition** every *N* frames |
| `--min-face` | `40` | Ignore faces smaller than this (px) at full resolution |
| `--detection-scale` | `0.5` | Resize factor for **detection** only (faster; boxes mapped back to full resolution) |

### Keyboard

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Cycle recognition model: VGG-Face → Facenet → ArcFace |
| `s` | Save screenshot (project root) |
| `a` | Register **largest** detected face: prompts for name in the terminal, saves crop under `known_faces/<name>/`, clears DeepFace’s `representations*.pkl` cache |

## Configuration

Edit `config.py` for default paths, per-model **recognition** distance thresholds (aligned with DeepFace verification), and `DEEPFACE_ALIGN` (cropped faces default to `False`).

## Architecture

Pipeline order: **detect** → crop regions → **recognize** each crop against the gallery.

- **`main.py`** — `VideoCapture` loop, frame skipping for recognition, HUD, keyboard UI.
- **`detector.py`** — OpenCV DNN face **detector** + confidence; optional auto-download of prototxt/caffemodel.
- **`recognizer.py`** — Background thread; **`DeepFace.find()`** on each cropped face; clears gallery pickle cache when identities change.
- **`utils.py`** — FPS, scaling boxes to full resolution, crop/draw helpers.
- **`config.py`** — Thresholds and constants.

**Detection** runs on **every frame** (optionally downscaled). **Recognition** runs **every N frames** in a **worker thread** so the preview stays responsive. Labels are held until the next consistent multi-face result so the UI does not flicker when async results lag by one frame.

## Notes

- **GPU**: TensorFlow/DeepFace will use GPU if configured in your environment; otherwise CPU.
- **Empty gallery**: Detection still finds faces; **recognition** labels them all **Unknown** until you add references under `known_faces/`.
- **Performance**: Lower `--detection-scale`, increase `--recognition-interval`, or use a lighter DeepFace model to improve FPS.
