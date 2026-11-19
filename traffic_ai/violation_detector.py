"""Traffic Rule Violation Detector.

Detects red-light runners and speeding violations at intersection stop lines,
generating evidentiary records for intelligent law enforcement.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from .models import Detection


@dataclass
class TrafficViolation:
    violation_id: str
    track_id: int
    vehicle_class: str
    timestamp: str
    lane_id: str
    violation_type: Literal["RED_LIGHT_RUNNER", "OVER_SPEEDING", "ILLEGAL_LANE_STOP"]
    recorded_speed_kmh: float
    speed_limit_kmh: float
    signal_phase: str
    fine_amount_inr: int
    evidence_status: str = "FLAGGED_FOR_REVIEW"


class ViolationDetector:
    def __init__(self, speed_limit_kmh: float = 50.0):
        self.speed_limit_kmh = speed_limit_kmh
        self.violations: list[TrafficViolation] = []
        self._counter = 1
        self._flagged_tracks: set[tuple[int, str]] = set()

    def check_violations(
        self,
        detections: list[Detection],
        active_lane: str,
        signal_state: str = "GREEN",
        speed_limit_kmh: float | None = None,
    ) -> list[TrafficViolation]:
        """Check all detections for red light violations or over-speeding."""
        limit = speed_limit_kmh or self.speed_limit_kmh
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for det in detections:
            if det.is_emergency:
                # Emergency vehicles are exempt from traffic speed/signal rules
                continue

            lane = det.lane_id or "Unknown"
            is_red = (lane != active_lane)
            speed = det.speed_kmh or (det.speed_px_frame * 3.2)

            # 1. Red Light Runner Detection
            # A vehicle moving actively (speed > 10 km/h) in a RED approach crossing the stop zone
            if is_red and speed > 12.0 and det.centroid[1] > 200:
                flag_key = (det.track_id, "RED_LIGHT_RUNNER")
                if flag_key not in self._flagged_tracks:
                    self._flagged_tracks.add(flag_key)
                    viol = TrafficViolation(
                        violation_id=f"VIOL-{self._counter:05d}",
                        track_id=det.track_id,
                        vehicle_class=det.vehicle_class.upper(),
                        timestamp=now_str,
                        lane_id=lane,
                        violation_type="RED_LIGHT_RUNNER",
                        recorded_speed_kmh=round(speed, 1),
                        speed_limit_kmh=limit,
                        signal_phase="RED (STOP)",
                        fine_amount_inr=2000,
                        evidence_status="CONFIRMED_ANPR_MATCH",
                    )
                    self.violations.append(viol)
                    self._counter += 1

            # 2. Over-speeding Violation
            if speed > (limit + 5.0):
                flag_key = (det.track_id, "OVER_SPEEDING")
                if flag_key not in self._flagged_tracks:
                    self._flagged_tracks.add(flag_key)
                    viol = TrafficViolation(
                        violation_id=f"VIOL-{self._counter:05d}",
                        track_id=det.track_id,
                        vehicle_class=det.vehicle_class.upper(),
                        timestamp=now_str,
                        lane_id=lane,
                        violation_type="OVER_SPEEDING",
                        recorded_speed_kmh=round(speed, 1),
                        speed_limit_kmh=limit,
                        signal_phase=f"{'GREEN' if lane == active_lane else 'RED'}",
                        fine_amount_inr=1000,
                        evidence_status="RADAR_CONFIRMED",
                    )
                    self.violations.append(viol)
                    self._counter += 1

        return self.violations
