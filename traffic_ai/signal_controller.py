"""Fixed-time and density-aware signal controllers."""

from .models import SignalDecision


class FixedTimeController:
    name = "Fixed-time"

    def __init__(self, green_seconds: int = 30):
        self.green_seconds = green_seconds

    def choose(self, queues: dict[str, float]) -> tuple[str, int]:
        lanes = list(queues)
        index = getattr(self, "_index", -1) + 1
        self._index = index % max(len(lanes), 1)
        return lanes[self._index], self.green_seconds


class RuleBasedController:
    name = "Rule-based"

    def __init__(self, base_time: int = 15, k: float = 4.0, min_time: int = 15, max_time: int = 60):
        self.base_time, self.k = base_time, k
        self.min_time, self.max_time = min_time, max_time

    def choose(self, queues: dict[str, float]) -> tuple[str, int]:
        lane_id = max(queues, key=queues.get)
        seconds = round(self.base_time + self.k * queues[lane_id])
        return lane_id, max(self.min_time, min(self.max_time, seconds))

    def decide(self, lane_metrics) -> SignalDecision:
        queues = {metric.lane_id: metric.weighted_density for metric in lane_metrics}
        lane_id, seconds = self.choose(queues)
        return SignalDecision(lane_id, seconds, f"Highest weighted density ({queues[lane_id]:.1f})", self.name)
