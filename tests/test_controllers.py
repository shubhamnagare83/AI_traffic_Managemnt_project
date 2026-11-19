"""Unit tests for Traffic Signal Controllers."""

import pytest
from traffic_ai.signal_controller import FixedTimeController, RuleBasedController, WebsterController
from traffic_ai.deep_marl_controller import DeepMARLController
from traffic_ai.models import LaneMetrics


def test_fixed_time_rotation():
    controller = FixedTimeController(green_seconds=25)
    queues = {"North": 10.0, "East": 5.0, "South": 2.0, "West": 8.0}
    l1, g1 = controller.choose(queues)
    l2, g2 = controller.choose(queues)
    assert g1 == 25
    assert g2 == 25
    assert l1 != l2


def test_webster_controller():
    controller = WebsterController()
    queues = {"North": 12.0, "East": 4.0, "South": 2.0, "West": 6.0}
    chosen_lane, green_sec = controller.choose(queues)
    assert chosen_lane == "North"
    assert 12 <= green_sec <= 65


def test_deep_marl_emergency_preemption():
    controller = DeepMARLController()
    metrics = [
        LaneMetrics("North", 10, 15.0, 20.0, 5, 80.0, 20.0),
        LaneMetrics("East", 2, 2.0, 5.0, 0, 15.0, 5.0),
        LaneMetrics("South", 3, 3.0, 8.0, 1, 20.0, 6.0),
        LaneMetrics("West", 4, 4.0, 10.0, 1, 25.0, 8.0),
    ]
    # Normal operation
    dec = controller.decide(metrics, evp_active=False)
    assert dec.lane_id == "North"

    # Emergency on East
    dec_evp = controller.decide(metrics, evp_active=True, evp_lane="East")
    assert dec_evp.lane_id == "East"
    assert dec_evp.evp_preemption is True
