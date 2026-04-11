"""Face detection using OpenCV DNN (SSD ResNet) with optional auto-download of weights."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import DETECTION_CONFIDENCE_THRESHOLD, MODELS_DIR

PROTOTXT_NAME = "deploy.prototxt"
CAFFEMODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/"
    + PROTOTXT_NAME
)
CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


def _ensure_model_files() -> tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    proto = MODELS_DIR / PROTOTXT_NAME
    weights = MODELS_DIR / CAFFEMODEL_NAME
    if not proto.is_file():
        urllib.request.urlretrieve(PROTOTXT_URL, proto)
    if not weights.is_file():
        urllib.request.urlretrieve(CAFFEMODEL_URL, weights)
    return proto, weights


class FaceDetector:
    """
    OpenCV DNN face detector.
    Returns boxes in the coordinate system of the *input image passed to detect*.
    """

    def __init__(self, confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold
        proto, weights = _ensure_model_files()
        self._net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

    def detect(
        self,
        image_bgr: np.ndarray,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in BGR image.
        Returns list of ((x, y, w, h), confidence) in pixel coordinates of `image_bgr`.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        results: List[Tuple[Tuple[int, int, int, int], float]] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            results.append(((x1, y1, bw, bh), conf))
        return results
