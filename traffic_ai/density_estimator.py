"""Lane ROI assignment and traffic density / queue metrics."""

from dataclasses import dataclass

from .models import Detection, LaneMetrics

WEIGHTS = {"car": 1.0, "bus": 2.5, "truck": 2.5, "motorcycle": 0.5, "bicycle": 0.5}


@dataclass
class LaneROI:
    lane_id: str
    polygon: list[tuple[float, float]]
    stop_line_y: float


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = current
        x2, y2 = previous
        intersects = (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
        if intersects:
            inside = not inside
        previous = current
    return inside


class DensityEstimator:
    def __init__(self, lanes: list[LaneROI], frame_rate: float = 30.0, pixels_per_meter: float = 12.0):
        self.lanes = lanes
        self.frame_rate = frame_rate
        self.pixels_per_meter = pixels_per_meter

    def assign_lanes(self, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            detection.lane_id = next(
                (lane.lane_id for lane in self.lanes if point_in_polygon(detection.centroid, lane.polygon)),
                None,
            )
        return detections

    def estimate(self, detections: list[Detection]) -> list[LaneMetrics]:
        self.assign_lanes(detections)
        metrics = []
        for lane in self.lanes:
            current = [d for d in detections if d.lane_id == lane.lane_id]
            stationary = [d for d in current if d.speed_px_frame == 0.0]
            queue_length = 0.0
            if stationary:
                farthest = max(abs(d.centroid[1] - lane.stop_line_y) for d in stationary)
                queue_length = farthest / self.pixels_per_meter
            metrics.append(
                LaneMetrics(
                    lane_id=lane.lane_id,
                    vehicle_count=len(current),
                    weighted_density=round(sum(WEIGHTS[d.vehicle_class] for d in current), 2),
                    queue_length_m=round(queue_length, 1),
                    stationary_count=len(stationary),
                    occupancy_pct=round(min(100.0, sum(WEIGHTS[d.vehicle_class] for d in current) * 7.5), 1),
                    flow_veh_per_min=round(len(current) * 60 / max(self.frame_rate, 1), 1),
                )
            )
        return metrics
