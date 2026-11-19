"""Unit tests for Computer Vision HUD Annotator and Camera functions."""

import numpy as np
from traffic_ai.camera_vision import CyberHUDAnnotator
from traffic_ai.demo_data import LANES, make_demo_frame
from traffic_ai.models import SignalDecision, V2XTelemetry


def test_render_hud_signature_resilience():
    dummy_frame = np.zeros((800, 800, 3), dtype=np.uint8)
    dets, _ = make_demo_frame(0)
    decision = SignalDecision("North", 30, "Test", "Rule-Based")
    telemetry = V2XTelemetry(80.0, 50.0, False, None, 2, 140, 6.5)

    # 1. Standard call with speed_limit_kmh
    out1 = CyberHUDAnnotator.render_hud(
        dummy_frame,
        detections=dets,
        lanes=LANES,
        signal_decision=decision,
        telemetry=telemetry,
        fps=30.0,
        speed_limit_kmh=50.0,
    )
    assert out1.shape == (800, 800, 3)

    # 2. Call with unexpected kwargs (resilience test)
    out2 = CyberHUDAnnotator.render_hud(
        dummy_frame,
        detections=dets,
        lanes=LANES,
        signal_decision=decision,
        telemetry=telemetry,
        fps=30.0,
        speed_limit_kmh=60.0,
        speed_limit=60.0,
        future_param_xyz=123,
    )
    assert out2.shape == (800, 800, 3)
