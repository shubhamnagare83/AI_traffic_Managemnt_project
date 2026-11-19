"""Signal controllers: Fixed-time, Rule-based adaptive, and Webster's Optimal Delay."""

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
        return SignalDecision(
            lane_id,
            seconds,
            f"Highest weighted density ({queues[lane_id]:.1f})",
            self.name,
        )


class WebsterController:
    """Webster's Optimum Delay Controller (Civil/ITS Engineering Standard).

    Implements F.V. Webster's classic formula for minimum vehicle delay at isolated intersections:
        C_0 = (1.5 * L + 5) / (1 - Y)
        g_i = (y_i / Y) * (C_0 - L)
    """

    name = "Webster's Optimal Delay"

    def __init__(
        self,
        lost_time_per_phase: float = 3.5,
        saturation_flow_vph: float = 1800.0,
        min_green: int = 12,
        max_green: int = 65,
    ):
        self.lost_time_per_phase = lost_time_per_phase
        self.saturation_flow_vph = saturation_flow_vph
        self.min_green = min_green
        self.max_green = max_green

    def compute_cycle_and_greens(self, flow_rates_vph: dict[str, float]) -> tuple[int, dict[str, int]]:
        num_phases = max(len(flow_rates_vph), 1)
        total_lost_time_L = self.lost_time_per_phase * num_phases

        # Flow ratios y_i = q_i / s_i
        y_ratios = {lane: q / self.saturation_flow_vph for lane, q in flow_rates_vph.items()}
        Y = sum(y_ratios.values())
        # Clamp Y to avoid singularity (if Y >= 1, intersection is over-saturated)
        Y = min(0.92, max(0.1, Y))

        # Optimum cycle time C_0
        c_0 = (1.5 * total_lost_time_L + 5.0) / (1.0 - Y)
        c_0 = max(40, min(120, round(c_0)))

        # Effective green time distribution g_i = (y_i / Y) * (C_0 - L)
        available_green = max(10, c_0 - int(total_lost_time_L))
        greens = {}
        for lane, y in y_ratios.items():
            g = round((y / Y) * available_green)
            greens[lane] = max(self.min_green, min(self.max_green, g))

        return c_0, greens

    def choose(self, queues: dict[str, float]) -> tuple[str, int]:
        # Convert density queue units to estimated hourly flow rate
        flows = {lane: q * 120.0 for lane, q in queues.items()}
        _, greens = self.compute_cycle_and_greens(flows)
        highest_lane = max(queues, key=queues.get)
        return highest_lane, greens.get(highest_lane, 25)

    def decide(self, lane_metrics) -> SignalDecision:
        flows = {m.lane_id: m.flow_veh_per_min * 60.0 for m in lane_metrics}
        cycle, greens = self.compute_cycle_and_greens(flows)
        highest_lane = max(flows, key=flows.get)
        green = greens.get(highest_lane, 25)
        reason = (
            f"Webster's Optimum Delay: Optimum Cycle C0={cycle}s | "
            f"Phase {highest_lane} allocated {green}s based on flow ratio"
        )
        return SignalDecision(highest_lane, green, reason, self.name)
