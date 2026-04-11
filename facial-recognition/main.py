"""
Real-time facial detection (OpenCV DNN) and recognition (DeepFace) from webcam.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime

import cv2

import config as cfg
from config import (
    DEFAULT_MODEL,
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_INPUT_SCALE,
    MIN_FACE_SIZE,
    MODEL_BACKENDS,
    RECOGNITION_FRAME_INTERVAL,
    KNOWN_FACES_DIR,
)
from detector import FaceDetector
from recognizer import RecognitionWorker, clear_deepface_representations_cache
from utils import (
    FPSCounter,
    crop_face_bgr,
    draw_face_overlay,
    draw_hud,
    largest_face_box,
    scale_detections_to_original,
)


def _sanitize_person_name(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'[<>:"/\\|?*]', "_", raw)
    return raw or "person"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam face detection + DeepFace recognition")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_BACKENDS),
        help="DeepFace model backend",
    )
    p.add_argument(
        "--det-conf",
        type=float,
        default=DETECTION_CONFIDENCE_THRESHOLD,
        help="Minimum DNN detection confidence",
    )
    p.add_argument(
        "--recognition-interval",
        type=int,
        default=RECOGNITION_FRAME_INTERVAL,
        help="Run DeepFace recognition every N frames",
    )
    p.add_argument(
        "--min-face",
        type=int,
        default=MIN_FACE_SIZE,
        help="Minimum face width/height in pixels at full resolution",
    )
    p.add_argument(
        "--detection-scale",
        type=float,
        default=DETECTION_INPUT_SCALE,
        help="Downscale factor for detection (e.g. 0.5 = half size)",
    )
    return p.parse_args()


def cycle_model(current: str) -> str:
    models = list(MODEL_BACKENDS)
    i = models.index(current) if current in models else 0
    return models[(i + 1) % len(models)]


def register_face_from_frame(
    frame,
    boxes: list[tuple[int, int, int, int]],
) -> None:
    """Interactive registration: largest face, terminal name, save under known_faces/."""
    box = largest_face_box(boxes)
    if box is None:
        print("No face detected to register.")
        return
    crop = crop_face_bgr(frame, box)
    if crop.size == 0:
        print("Empty crop; try again.")
        return
    try:
        name = input("Enter name for this face: ")
    except EOFError:
        return
    name = _sanitize_person_name(name)
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    fname = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    out_path = person_dir / fname
    cv2.imwrite(str(out_path), crop)
    clear_deepface_representations_cache(KNOWN_FACES_DIR)
    print(f"Saved reference to {out_path}. DeepFace gallery cache cleared.")


def main() -> int:
    args = parse_args()
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(confidence_threshold=args.det_conf)
    worker = RecognitionWorker(db_path=KNOWN_FACES_DIR)
    worker.set_model(args.model)
    worker.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}", file=sys.stderr)
        worker.stop()
        worker.join(timeout=2.0)
        return 1

    fps_counter = FPSCounter()
    frame_index = 0
    window = "Real-Time Facial Recognition"
    rec_state: dict = {"labels": [], "n": -1}

    print("Controls: q quit | m cycle model | s screenshot | a add face (terminal name)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame grab failed; exiting.")
                break

            frame_index += 1
            fps = fps_counter.tick()
            h, w = frame.shape[:2]
            scale = float(args.detection_scale)
            if scale > 0 and abs(scale - 1.0) > 1e-6:
                det_frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                det_frame = frame

            raw_dets = detector.detect(det_frame)
            scaled_boxes = scale_detections_to_original(
                [b for b, _ in raw_dets], scale if scale > 0 else 1.0
            )
            det_confs = [c for _, c in raw_dets]

            faces_full: list[tuple[tuple[int, int, int, int], float]] = []
            for box, conf in zip(scaled_boxes, det_confs):
                bw, bh = box[2], box[3]
                if bw < args.min_face or bh < args.min_face:
                    continue
                faces_full.append((box, conf))

            faces_sorted = sorted(faces_full, key=lambda t: t[0][0])
            sorted_boxes = [b for b, _ in faces_sorted]
            sorted_confs = [c for _, c in faces_sorted]

            if len(sorted_boxes) != rec_state.get("n", -1):
                rec_state["labels"] = []
            rec_state["n"] = len(sorted_boxes)

            if (
                args.recognition_interval > 0
                and frame_index % args.recognition_interval == 0
                and sorted_boxes
            ):
                job = []
                for box in sorted_boxes:
                    job.append((box, crop_face_bgr(frame, box)))
                worker.submit(job)

            latest = worker.get_latest()
            if sorted_boxes and len(latest) == len(sorted_boxes):
                latest_sorted = sorted(latest, key=lambda r: r[0][0])
                rec_state["labels"] = [(n, p) for (_, n, p) in latest_sorted]

            if len(rec_state["labels"]) == len(sorted_boxes):
                labels = rec_state["labels"]
            else:
                labels = [("Unknown", None)] * len(sorted_boxes)

            display = frame.copy()
            for box, dconf, (name, pct) in zip(
                sorted_boxes, sorted_confs, labels
            ):
                recognized = name != "Unknown"
                draw_face_overlay(
                    display, box, name, pct, recognized, det_confidence=dconf
                )

            model_name = worker.get_model()
            draw_hud(
                display,
                fps,
                model_name,
                lines_extra=[
                    f"Det conf (min): {args.det_conf:.2f}",
                    f"Recog every {args.recognition_interval} frames",
                ],
            )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                new_model = cycle_model(model_name)
                worker.set_model(new_model)
                print(f"Switched model to: {new_model}")
            if key == ord("s"):
                shot = cfg.PROJECT_ROOT / datetime.now().strftime(
                    "screenshot_%Y%m%d_%H%M%S.jpg"
                )
                cv2.imwrite(str(shot), display)
                print(f"Saved {shot}")
            if key == ord("a"):
                register_face_from_frame(frame, sorted_boxes)

            time.sleep(0.001)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        worker.stop()
        worker.join(timeout=5.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
