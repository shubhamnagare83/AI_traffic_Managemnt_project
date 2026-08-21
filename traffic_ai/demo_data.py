"""Deterministic synthetic intersection data for demos and development."""

from .density_estimator import DensityEstimator, LaneROI
from .models import Detection


LANES = [
    LaneROI("North", [(150, 0), (350, 0), (350, 420), (150, 420)], 380),
    LaneROI("East", [(350, 150), (800, 150), (800, 350), (350, 350)], 380),
    LaneROI("South", [(450, 420), (650, 420), (650, 800), (450, 800)], 420),
    LaneROI("West", [(0, 450), (450, 450), (450, 650), (0, 650)], 420),
]


def make_demo_frame(frame_index: int, lane_counts: dict[str, int] | None = None) -> tuple[list[Detection], list]:
    counts = lane_counts or {"North": 9, "East": 5, "South": 3, "West": 7}
    detections = []
    track_id = 1
    for lane_index, lane in enumerate(LANES):
        count = counts.get(lane.lane_id, 0)
        for index in range(count):
            stationary = index % 3 != 0
            if lane.lane_id in ("North", "South"):
                x = 190 + (index % 3) * 65
                y = 355 - (index // 3) * 42
            else:
                x = 395 + (index // 3) * 42
                y = 185 + (index % 3) * 58
            if not stationary:
                x += (frame_index % 9) * 2
            vehicle_class = ("car", "car", "truck", "motorcycle")[index % 4]
            detections.append(Detection(track_id, vehicle_class, 0.88, (x - 15, y - 10, x + 15, y + 10), (x, y), speed_px_frame=0 if stationary else 4.0))
            track_id += 1
    estimator = DensityEstimator(LANES)
    return detections, estimator.estimate(detections)
