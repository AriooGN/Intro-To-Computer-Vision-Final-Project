"""Application configuration: models, thresholds, and paths."""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

KNOWN_FACES_DIR = PROJECT_ROOT / "known_faces"
MODELS_DIR = PROJECT_ROOT / "models"

# Supported DeepFace backends (cycle with 'm')
MODEL_BACKENDS = ("VGG-Face", "Facenet", "ArcFace")
DEFAULT_MODEL = MODEL_BACKENDS[0]

# Recognition distance thresholds (same metric DeepFace uses per model in verification)
RECOGNITION_THRESHOLDS = {
    "VGG-Face": 0.40,
    "Facenet": 0.40,
    "ArcFace": 0.68,
}

# OpenCV DNN face detector minimum confidence (0–1)
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Run DeepFace recognition every N frames (detection still every frame)
RECOGNITION_FRAME_INTERVAL = 5

# Ignore detections smaller than this in the original (full-res) frame (pixels)
MIN_FACE_SIZE = 40

# Scale factor for detection branch (1.0 = full resolution)
DETECTION_INPUT_SCALE = 0.5

# DeepFace.find alignment (cropped faces from our detector)
DEEPFACE_ALIGN = False
