"""FPS tracking, geometry, and drawing helpers."""

from __future__ import annotations

import time
from collections import deque
from typing import Tuple

import cv2
import numpy as np


class FPSCounter:
    """Smooth FPS estimate from inter-frame intervals."""

    def __init__(self, window_size: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window_size)
        self._last_tick = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._last_tick)
        self._last_tick = now
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)


def scale_detections_to_original(
    boxes: list[Tuple[int, int, int, int]],
    scale: float,
) -> list[Tuple[int, int, int, int]]:
    """Map boxes from scaled detection space back to original frame coordinates."""
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return boxes
    inv = 1.0 / scale
    out = []
    for x, y, w, h in boxes:
        out.append(
            (
                int(round(x * inv)),
                int(round(y * inv)),
                int(round(w * inv)),
                int(round(h * inv)),
            )
        )
    return out


def crop_face_bgr(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop BGR region; clamps to frame bounds."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y0:y1, x0:x1].copy()


def draw_face_overlay(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence_pct: float | None,
    recognized: bool,
    det_confidence: float | None = None,
    font_scale: float = 0.55,
) -> None:
    """Draw bounding box and label on frame (BGR, in-place)."""
    x, y, bw, bh = box
    color = (0, 200, 0) if recognized else (0, 0, 255)
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, thickness, cv2.LINE_AA)

    parts = [label]
    if confidence_pct is not None:
        parts.append(f"{confidence_pct:.1f}%")
    if det_confidence is not None:
        parts.append(f"det {det_confidence * 100:.0f}%")
    text = " | ".join(parts).strip()
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    ty = max(y - 4, th + 6)
    cv2.rectangle(
        frame,
        (x, ty - th - 6),
        (x + tw + 6, ty + baseline),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, ty - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    model_name: str,
    lines_extra: list[str] | None = None,
) -> None:
    """Corner HUD: FPS and active model."""
    h, w = frame.shape[:2]
    y0 = 24
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Model: {model_name}",
        (10, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if lines_extra:
        for i, line in enumerate(lines_extra):
            cv2.putText(
                frame,
                line,
                (10, y0 + 56 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )


def largest_face_box(
    boxes: list[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])
