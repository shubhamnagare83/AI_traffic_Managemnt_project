from dataclasses import dataclass, field
from typing import Literal

VehicleClass = Literal[
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "autonomous_pod",
    "emergency",
    "ev_shuttle",
]


@dataclass
class Detection:
    track_id: int
    vehicle_class: VehicleClass
    confidence: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    lane_id: str | None = None
    speed_px_frame: float = 0.0
    speed_kmh: float = 0.0
    is_cav: bool = False
    is_emergency: bool = False
    glosa_speed_kmh: float = 50.0
    v2x_id: str | None = None


@dataclass
class LaneMetrics:
    lane_id: str
    vehicle_count: int
    weighted_density: float
    queue_length_m: float
    stationary_count: int
    occupancy_pct: float
    flow_veh_per_min: float
    cav_count: int = 0
    emergency_present: bool = False


@dataclass
class SignalDecision:
    lane_id: str
    green_seconds: int
    reason: str
    controller: str
    evp_preemption: bool = False
    confidence_score: float = 0.98


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    lanes: list[LaneMetrics] = field(default_factory=list)


@dataclass
class IncidentAlert:
    alert_id: str
    timestamp: str
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    alert_type: str
    lane_id: str
    description: str
    autonomous_action: str


@dataclass
class V2XTelemetry:
    connected_ratio_pct: float
    glosa_advisory_kmh: float
    evp_active: bool
    evp_lane: str | None
    platoons_active: int
    v2x_packets_per_sec: int
    network_latency_ms: float


@dataclass
class EcoMetrics:
    co2_saved_kg: float
    fuel_saved_liters: float
    ev_energy_saved_kwh: float
    idle_hours_reduced: float
    noise_reduction_db: float
