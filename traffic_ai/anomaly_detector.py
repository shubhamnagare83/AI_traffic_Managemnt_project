"""2030 AI Anomaly, Safety & Incident Radar."""

from datetime import datetime
from .models import Detection, IncidentAlert, LaneMetrics


class AnomalyDetector:
    def __init__(self):
        self.history: list[IncidentAlert] = []
        self._alert_counter = 1

    def scan(
        self,
        detections: list[Detection],
        lane_metrics: list[LaneMetrics],
        active_lane: str,
        evp_active: bool = False,
    ) -> list[IncidentAlert]:
        alerts: list[IncidentAlert] = []
        now_str = datetime.now().strftime("%H:%M:%S")

        # 1. Stalled vehicle in active green corridor
        for det in detections:
            if det.lane_id == active_lane and det.speed_kmh < 1.0 and det.speed_px_frame == 0:
                alert = IncidentAlert(
                    alert_id=f"INC-{self._alert_counter:04d}",
                    timestamp=now_str,
                    severity="WARNING",
                    alert_type="Stalled Vehicle in Green Corridor",
                    lane_id=active_lane,
                    description=f"Track #{det.track_id} ({det.vehicle_class}) stationary in active approach {active_lane}.",
                    autonomous_action="V2X advisory broadcasted: Approaching CAVs rerouting around obstacle.",
                )
                alerts.append(alert)
                self._alert_counter += 1
                break

        # 2. Critical queue overflow risk
        for metric in lane_metrics:
            if metric.queue_length_m >= 35.0:
                alert = IncidentAlert(
                    alert_id=f"INC-{self._alert_counter:04d}",
                    timestamp=now_str,
                    severity="CRITICAL",
                    alert_type="Severe Queue Saturation",
                    lane_id=metric.lane_id,
                    description=f"Approach {metric.lane_id} queue exceeded 35m ({metric.queue_length_m}m). Gridlock hazard.",
                    autonomous_action="Deep-MARL extending phase allocation + upstream perimeter metering engaged.",
                )
                alerts.append(alert)
                self._alert_counter += 1

        # 3. Emergency Vehicle Priority Alert
        if evp_active:
            alert = IncidentAlert(
                alert_id=f"INC-{self._alert_counter:04d}",
                timestamp=now_str,
                severity="CRITICAL",
                alert_type="Active Emergency Preemption Corridor",
                lane_id=active_lane,
                description=f"Emergency service vehicle inbound on {active_lane}. Cross-traffic holding red.",
                autonomous_action="EVP corridor green wave cleared. Digital road studs flashing amber.",
            )
            alerts.append(alert)
            self._alert_counter += 1

        # Keep a rolling buffer of latest 25 alerts
        for alert in alerts:
            # Avoid duplicate alert types for same lane within short window
            if not any(h.alert_type == alert.alert_type and h.lane_id == alert.lane_id for h in self.history[-4:]):
                self.history.append(alert)

        self.history = self.history[-25:]
        return self.history
