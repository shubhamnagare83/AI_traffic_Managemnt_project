"""Lightweight intersection simulator driven by extracted lane densities."""

from dataclasses import dataclass
import random


@dataclass
class SimulationResult:
    controller: str
    average_wait_seconds: float
    throughput: int
    max_queue: float
    total_wait_seconds: float


class IntersectionSimulator:
    def __init__(self, seed: int = 7):
        self.random = random.Random(seed)

    def run(self, controller, lane_inputs: dict[str, float], duration_seconds: int = 900) -> SimulationResult:
        queues = {lane: max(1.0, density * self.random.uniform(0.9, 1.15)) for lane, density in lane_inputs.items()}
        total_wait = 0.0
        throughput = 0
        max_queue = max(queues.values(), default=0.0)
        clock = 0
        while clock < duration_seconds:
            lane_id, green_time = controller.choose(queues)
            for lane in queues:
                arrivals = lane_inputs[lane] * 0.018 * green_time * self.random.uniform(0.85, 1.15)
                discharge = green_time * 0.10 if lane == lane_id else green_time * 0.012
                queues[lane] = max(0.0, queues[lane] + arrivals - discharge)
                if lane == lane_id:
                    throughput += int(min(discharge, queues[lane] + discharge))
            total_wait += sum(queues.values()) * green_time
            max_queue = max(max_queue, max(queues.values(), default=0.0))
            clock += green_time
        vehicles = max(throughput, 1)
        return SimulationResult(controller.name, round(total_wait / vehicles, 1), throughput, round(max_queue, 1), round(total_wait, 1))
