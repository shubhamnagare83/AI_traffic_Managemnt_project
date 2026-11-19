"""2030 Connected & Autonomous Vehicle (V2X) and Emergency Preemption System."""

from typing import Iterable
from .models import Detection, SignalDecision, V2XTelemetry


class V2XSystem:
    def __init__(self, target_speed_kmh: float = 50.0):
        self.target_speed_kmh = target_speed_kmh
        self.evp_active = False
        self.evp_lane: str | None = None
        self.evp_timer_sec: int = 0

    def trigger_emergency_preemption(self, lane_id: str, duration_sec: int = 35) -> None:
        """Trigger emergency corridor preemption (EVP) for inbound emergency vehicles."""
        self.evp_active = True
        self.evp_lane = lane_id
        self.evp_timer_sec = duration_sec

    def cancel_emergency_preemption(self) -> None:
        """Clear emergency corridor status."""
        self.evp_active = False
        self.evp_lane = None
        self.evp_timer_sec = 0

    def compute_glosa_advisory(
        self,
        distance_to_stop_m: float,
        time_to_green_sec: float,
        current_phase_green: bool,
    ) -> float:
        """Calculate Green Light Optimal Speed Advisory (GLOSA) in km/h.

        Directs approaching autonomous vehicles to adjust speed so they cruise through
        the green light without coming to a stop.
        """
        if current_phase_green:
            if time_to_green_sec <= 2.0:
                # Approaching end of green - advice to maintain safe cruise
                return max(30.0, min(60.0, self.target_speed_kmh))
            # Optimal speed to clear the intersection smoothly
            return max(35.0, min(55.0, self.target_speed_kmh))

        # Currently red: compute speed to arrive exactly as light turns green
        if time_to_green_sec <= 0.5:
            return self.target_speed_kmh

        required_mps = distance_to_stop_m / max(time_to_green_sec, 1.0)
        advisory_kmh = required_mps * 3.6
        # Clamp between safe city bounds [25 km/h, 60 km/h]
        return round(max(25.0, min(60.0, advisory_kmh)), 1)

    def process_detections(
        self,
        detections: list[Detection],
        active_lane: str,
        remaining_green_sec: int,
        cav_ratio_override: float = 0.82,
    ) -> tuple[list[Detection], V2XTelemetry]:
        """Tag detections with V2X connected status, speed, and GLOSA speed advisory."""
        emergency_detected = False
        emergency_lane = self.evp_lane

        # Check for emergency vehicles in detections
        for det in detections:
            if det.vehicle_class == "emergency" or det.is_emergency:
                emergency_detected = True
                if det.lane_id:
                    emergency_lane = det.lane_id

        if emergency_detected and emergency_lane and not self.evp_active:
            self.trigger_emergency_preemption(emergency_lane)

        # Count CAVs and platoons
        cav_count = 0
        for i, det in enumerate(detections):
            # Deterministic CAV assignment if not explicitly set
            if not hasattr(det, "is_cav") or not det.is_cav:
                det.is_cav = (det.track_id * 17) % 100 < (cav_ratio_override * 100)
            if det.is_cav:
                cav_count += 1
                det.v2x_id = f"CAV-{det.track_id:04d}"

            # Calculate speed in km/h
            if getattr(det, "speed_kmh", 0.0) == 0.0:
                det.speed_kmh = round(det.speed_px_frame * 3.2, 1) if det.speed_px_frame > 0 else 0.0

            # Calculate GLOSA advisory
            is_green = (det.lane_id == active_lane)
            time_to_green = 0.0 if is_green else max(5.0, float(remaining_green_sec))
            dist_to_stop = 80.0  # approximate meters from stop line
            det.glosa_speed_kmh = self.compute_glosa_advisory(dist_to_stop, time_to_green, is_green)

        total_vehicles = max(len(detections), 1)
        connected_pct = round((cav_count / total_vehicles) * 100, 1)
        platoons = max(1, cav_count // 3)

        telemetry = V2XTelemetry(
            connected_ratio_pct=connected_pct,
            glosa_advisory_kmh=round(self.target_speed_kmh, 1),
            evp_active=self.evp_active,
            evp_lane=self.evp_lane,
            platoons_active=platoons,
            v2x_packets_per_sec=max(120, cav_count * 28),
            network_latency_ms=round(6.5 + (total_vehicles % 4) * 0.8, 1),
        )
        return detections, telemetry
