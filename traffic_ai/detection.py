"""YOLOv8 vehicle detection with an optional deterministic demo source."""

from pathlib import Path
from typing import Iterable

from .models import Detection, VehicleClass

VEHICLE_CLASSES: dict[int, VehicleClass] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YOLODetector:
    """Thin wrapper around Ultralytics YOLOv8n.

    The model is imported lazily so the dashboard can run in demo mode on a
    clean machine. On first real-video inference Ultralytics downloads weights.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO mode needs ultralytics. Install requirements.txt first."
            ) from exc
        self.model = YOLO(model_path)
        self.confidence = confidence

    def track(self, frame, frame_index: int = 0) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES),
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        detections: list[Detection] = []
        for index in range(len(boxes)):
            cls_id = int(boxes.cls[index].item())
            if cls_id not in VEHICLE_CLASSES:
                continue
            coords = boxes.xyxy[index].tolist()
            x1, y1, x2, y2 = (int(value) for value in coords)
            track_id = (
                int(boxes.id[index].item())
                if boxes.id is not None
                else frame_index * 10000 + index
            )
            detections.append(
                Detection(
                    track_id=track_id,
                    vehicle_class=VEHICLE_CLASSES[cls_id],
                    confidence=float(boxes.conf[index].item()),
                    bbox=(x1, y1, x2, y2),
                    centroid=((x1 + x2) / 2, (y1 + y2) / 2),
                )
            )
        return detections


def open_video(path: str):
    """Open an MP4 source, raising a useful error instead of failing silently."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video mode needs opencv-python-headless.") from exc
    if not Path(path).exists():
        raise FileNotFoundError(f"Video not found: {path}")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    return capture


def iter_video(path: str, detector: YOLODetector, max_frames: int = 300) -> Iterable[tuple[object, list[Detection]]]:
    capture = open_video(path)
    try:
        for frame_index in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            yield frame, detector.track(frame, frame_index)
    finally:
        capture.release()
