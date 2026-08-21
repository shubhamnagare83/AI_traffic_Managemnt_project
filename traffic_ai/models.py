from dataclasses import dataclass, field
from typing import Literal


VehicleClass = Literal["car", "bus", "truck", "motorcycle", "bicycle"]


@dataclass
class Detection:
    track_id: int
    vehicle_class: VehicleClass
    confidence: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    lane_id: str | None = None
    speed_px_frame: float = 0.0


@dataclass
class LaneMetrics:
    lane_id: str
    vehicle_count: int
    weighted_density: float
    queue_length_m: float
    stationary_count: int
    occupancy_pct: float
    flow_veh_per_min: float


@dataclass
class SignalDecision:
    lane_id: str
    green_seconds: int
    reason: str
    controller: str


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    lanes: list[LaneMetrics] = field(default_factory=list)
