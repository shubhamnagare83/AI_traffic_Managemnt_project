"""Unit tests for V2X System and Violation Detection."""

from traffic_ai.models import Detection
from traffic_ai.v2x_system import V2XSystem
from traffic_ai.violation_detector import ViolationDetector


def test_v2x_glosa():
    v2x = V2XSystem(target_speed_kmh=50.0)
    # When approaching green light
    spd_green = v2x.compute_glosa_advisory(80.0, 10.0, current_phase_green=True)
    assert 30.0 <= spd_green <= 60.0

    # When light is red and turns green in 6 seconds
    spd_red = v2x.compute_glosa_advisory(80.0, 6.0, current_phase_green=False)
    assert 25.0 <= spd_red <= 60.0


def test_violation_detector():
    detector = ViolationDetector(speed_limit_kmh=50.0)
    dets = [
        # Normal vehicle
        Detection(1, "car", 0.9, (10, 10, 30, 30), (20.0, 20.0), lane_id="North", speed_kmh=42.0),
        # Speeding vehicle
        Detection(2, "car", 0.9, (10, 10, 30, 30), (20.0, 20.0), lane_id="North", speed_kmh=78.0),
        # Red light runner on East (when active lane is North)
        Detection(3, "truck", 0.9, (10, 220, 30, 240), (20.0, 230.0), lane_id="East", speed_kmh=35.0, speed_px_frame=5.0),
    ]

    violations = detector.check_violations(dets, active_lane="North")
    assert len(violations) >= 2
    types = [v.violation_type for v in violations]
    assert "OVER_SPEEDING" in types
    assert "RED_LIGHT_RUNNER" in types
