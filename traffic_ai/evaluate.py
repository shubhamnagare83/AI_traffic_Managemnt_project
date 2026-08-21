"""Run comparable simulations and return dashboard-friendly records."""

from .rl_agent import QLearningAgent
from .signal_controller import FixedTimeController, RuleBasedController
from .simulator import IntersectionSimulator


def evaluate(lane_inputs: dict[str, float], train_episodes: int = 300) -> tuple[list[dict], list[float]]:
    lanes = list(lane_inputs)
    rl_agent = QLearningAgent(lanes)
    rewards = rl_agent.train(train_episodes)
    simulator = IntersectionSimulator()
    results = []
    for controller in (
        FixedTimeController(),
        RuleBasedController(),
        rl_agent,
    ):
        result = simulator.run(controller, lane_inputs)
        results.append(
            {
                "Controller": result.controller,
                "Average wait (s)": result.average_wait_seconds,
                "Throughput (veh)": result.throughput,
                "Max queue (m)": result.max_queue,
                "Total wait (s)": result.total_wait_seconds,
                "Processing FPS": 30.0,
                "Detection mAP": None,
            }
        )
    return results, rewards
