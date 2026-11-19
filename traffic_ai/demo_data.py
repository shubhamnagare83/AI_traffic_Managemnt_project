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

# Default active layout
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
) -> tuple[list[Detection], list]:
    current_lanes = lanes_preset or LANES
    counts = lane_counts or {"North": 9, "East": 5, "South": 3, "West": 7}
    detections = []
    track_id = 1

    for lane in current_lanes:
        count = counts.get(lane.lane_id, 4)
        for index in range(count):
            stationary = (index % 3 != 0)
            poly = lane.polygon
            min_x = min(p[0] for p in poly)
            max_x = max(p[0] for p in poly)
            min_y = min(p[1] for p in poly)
            max_y = max(p[1] for p in poly)

            # Distribute vehicles realistically inside the lane polygon
            frac = (index + 1) / max(count + 1, 2)
            if (max_y - min_y) >= (max_x - min_x):
                # Vertical road orientation
                x = min_x + 35 + (index % 2) * 45
                y = lane.stop_line_y - (index + 1) * 38 if lane.stop_line_y < 500 else lane.stop_line_y + (index + 1) * 38
                y = max(min_y + 20, min(max_y - 20, y))
            else:
                # Horizontal road orientation
                y = min_y + 35 + (index % 2) * 45
                x = lane.stop_line_y - (index + 1) * 38 if lane.stop_line_y < 500 else lane.stop_line_y + (index + 1) * 38
                x = max(min_x + 20, min(max_x - 20, x))

            if not stationary:
                x += (frame_index % 7) * 2

            vehicle_class = ("car", "car", "bus", "motorcycle", "truck")[index % 5]
            speed = 0.0 if stationary else round(35.0 + (index % 4) * 6.0, 1)

            detections.append(
                Detection(
                    track_id=track_id,
                    vehicle_class=vehicle_class,
                    confidence=0.91,
                    bbox=(int(x - 18), int(y - 12), int(x + 18), int(y + 12)),
                    centroid=(float(x), float(y)),
                    lane_id=lane.lane_id,
                    speed_px_frame=0.0 if stationary else 4.2,
                    speed_kmh=speed,
                    is_cav=(track_id % 4 != 0),
                )
            )
            track_id += 1

    estimator = DensityEstimator(current_lanes)
    return detections, estimator.estimate(detections)
