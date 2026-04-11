"""DeepFace-based identity lookup with background recognition and DB cache refresh."""

from __future__ import annotations

import glob
import queue
import threading
import tempfile
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

from config import DEEPFACE_ALIGN, KNOWN_FACES_DIR, RECOGNITION_THRESHOLDS

BBox = Tuple[int, int, int, int]


def clear_deepface_representations_cache(db_path: Path) -> None:
    """
    Remove DeepFace's cached representations under the gallery folder so the next
    `DeepFace.find()` rebuilds embeddings after images change.
    """
    pattern = str(db_path / "representations*.pkl")
    for p in glob.glob(pattern):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def _distance_from_row(row: Any) -> float:
    """Extract numeric distance / similarity column from DeepFace.find result row."""
    # DeepFace uses columns like 'VGG-Face_cosine', 'ArcFace_cosine', etc.
    for col in row.index:
        if col == "identity":
            continue
        val = row[col]
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 1.0


def _identity_name_from_path(identity_path: str) -> str:
    return Path(identity_path).parent.name


def match_crop_with_deepface_find(
    crop_bgr: np.ndarray,
    db_path: Path,
    model_name: str,
    threshold: float,
) -> Tuple[str, float | None]:
    """
    Run DeepFace.find on a single cropped face and return (label, confidence_percent).
    Uses the same distance threshold semantics as the rest of the app.
    """
    from deepface import DeepFace  # lazy import for faster startup

    if crop_bgr.size == 0 or max(crop_bgr.shape[:2]) < 8:
        return "Unknown", None

    db_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, crop_bgr)
        dfs = DeepFace.find(
            img_path=tmp_path,
            db_path=str(db_path),
            model_name=model_name,
            enforce_detection=False,
            align=DEEPFACE_ALIGN,
            silent=True,
        )
    except Exception:
        return "Unknown", None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not dfs or dfs[0] is None or len(dfs[0]) == 0:
        return "Unknown", None

    df = dfs[0]
    row = df.iloc[0]
    distance = _distance_from_row(row)
    if distance > threshold:
        return "Unknown", None

    name = _identity_name_from_path(str(row["identity"]))
    # Map distance to a simple "confidence" display: 100% at 0, 0% at threshold
    conf_pct = max(0.0, min(100.0, (1.0 - distance / threshold) * 100.0))
    return name, conf_pct


class RecognitionWorker(threading.Thread):
    """
    Consumes recognition jobs in a background thread so the webcam loop stays smooth.
    Each job is a list of (bbox, crop_bgr) already sorted for stable ordering.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(daemon=True)
        self._db_path = db_path or KNOWN_FACES_DIR
        self._queue: queue.Queue[
            tuple[list[tuple[BBox, np.ndarray]], str] | None
        ] = queue.Queue(maxsize=2)
        self._model_name = "VGG-Face"
        self._threshold = RECOGNITION_THRESHOLDS.get(self._model_name, 0.4)
        self._lock = threading.Lock()
        self._latest: list[tuple[BBox, str, float | None]] = []
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()

    def set_model(self, model_name: str) -> None:
        with self._lock:
            self._model_name = model_name
            self._threshold = RECOGNITION_THRESHOLDS.get(model_name, 0.4)
            clear_deepface_representations_cache(self._db_path)

    def get_model(self) -> str:
        with self._lock:
            return self._model_name

    def get_threshold(self) -> float:
        with self._lock:
            return self._threshold

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

    def submit(self, faces: list[tuple[BBox, np.ndarray]]) -> None:
        """Non-blocking: drop backlog if the worker is still busy."""
        with self._lock:
            model = self._model_name
        job = (faces, model)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.put_nowait(job)
            except Exception:
                pass

    def get_latest(self) -> list[tuple[BBox, str, float | None]]:
        with self._latest_lock:
            return list(self._latest)

    def run(self) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            if job is None:
                break
            faces, model_name = job
            results: list[tuple[BBox, str, float | None]] = []
            threshold = RECOGNITION_THRESHOLDS.get(model_name, 0.4)
            for bbox, crop in faces:
                label, conf = match_crop_with_deepface_find(
                    crop, self._db_path, model_name, threshold
                )
                results.append((bbox, label, conf))
            with self._latest_lock:
                self._latest = results
