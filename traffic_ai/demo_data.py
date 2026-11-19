"""Configurable Road Layouts, Intersection Polygons, and Deterministic Traffic Data."""

from .density_estimator import DensityEstimator, LaneROI
from .models import Detection

# 1. Standard 4-Way Cross Intersection
LANES_4WAY = [
    LaneROI("North", [(250, 0), (550, 0), (550, 245), (250, 245)], stop_line_y=245.0),
    LaneROI("East", [(555, 250), (800, 250), (800, 550), (555, 550)], stop_line_y=555.0),
    LaneROI("South", [(250, 555), (550, 555), (550, 800), (250, 800)], stop_line_y=555.0),
    LaneROI("West", [(0, 250), (245, 250), (245, 550), (0, 550)], stop_line_y=245.0),
]

# 2. T-Junction (North, East, West)
LANES_TJUNC = [
    LaneROI("North", [(250, 0), (550, 0), (550, 245), (250, 245)], stop_line_y=245.0),
    LaneROI("East", [(555, 250), (800, 250), (800, 550), (555, 550)], stop_line_y=555.0),
    LaneROI("West", [(0, 250), (245, 250), (245, 550), (0, 550)], stop_line_y=245.0),
]

# 3. Dual-Carriageway Arterial Road (Inbound vs Outbound)
LANES_HIGHWAY = [
    LaneROI("Inbound Northbound", [(150, 0), (380, 0), (380, 800), (150, 800)], stop_line_y=400.0),
    LaneROI("Outbound Southbound", [(420, 0), (650, 0), (650, 800), (420, 800)], stop_line_y=400.0),
]

LANES = LANES_4WAY

ROAD_PRESETS = {
    "4-Way Urban Intersection": LANES_4WAY,
    "T-Junction (3-Way)": LANES_TJUNC,
    "Dual-Carriageway Arterial Road": LANES_HIGHWAY,
}


def make_demo_frame(
    frame_index: int,
    lane_counts: dict[str, int] | None = None,
    lanes_preset: list[LaneROI] | None = None,
    active_green_lane: str = "North",
) -> tuple[list[Detection], list]:
    current_lanes = lanes_preset or LANES
    counts = lane_counts or {"North": 8, "East": 5, "South": 3, "West": 6}
    detections = []
    track_id = 1

    speed_samples = [36.0, 42.5, 48.0, 54.5, 32.0, 62.0]

    for lane in current_lanes:
        count = counts.get(lane.lane_id, 4)
        is_lane_green = (lane.lane_id == active_green_lane)

        for index in range(count):
            poly = lane.polygon
            min_x = min(p[0] for p in poly)
            max_x = max(p[0] for p in poly)
            min_y = min(p[1] for p in poly)
            max_y = max(p[1] for p in poly)

            # In red lanes, vehicles near stop lines are stationary/queued
            if not is_lane_green:
                stationary = (index % 4 != 0)
            else:
                stationary = False

            if (max_y - min_y) >= (max_x - min_x):
                # Vertical road orientation
                x = min_x + 35 + (index % 2) * 50
                y = lane.stop_line_y - (index + 1) * 38 if lane.stop_line_y < 500 else lane.stop_line_y + (index + 1) * 38
                y = max(min_y + 20, min(max_y - 20, y))
            else:
                # Horizontal road orientation
                y = min_y + 35 + (index % 2) * 50
                x = lane.stop_line_y - (index + 1) * 38 if lane.stop_line_y < 500 else lane.stop_line_y + (index + 1) * 38
                x = max(min_x + 20, min(max_x - 20, x))

            if not stationary:
                speed = speed_samples[(track_id + index) % len(speed_samples)]
                speed_px = round(speed / 3.2, 2)
            else:
                speed = 0.0
                speed_px = 0.0

            vehicle_class = ("car", "car", "bus", "motorcycle", "truck")[index % 5]

            detections.append(
                Detection(
                    track_id=track_id,
                    vehicle_class=vehicle_class,
                    confidence=0.92,
                    bbox=(int(x - 18), int(y - 12), int(x + 18), int(y + 12)),
                    centroid=(float(x), float(y)),
                    lane_id=lane.lane_id,
                    speed_px_frame=speed_px,
                    speed_kmh=speed,
                    is_cav=(track_id % 3 != 0),
                )
            )
            track_id += 1

    estimator = DensityEstimator(current_lanes)
    return detections, estimator.estimate(detections)
