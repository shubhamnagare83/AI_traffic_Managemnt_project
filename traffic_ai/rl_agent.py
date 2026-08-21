"""Tabular Q-learning controller for discretized intersection states."""

from collections import defaultdict
import random


class QLearningAgent:
    name = "Q-learning"
    ACTIONS = (15, 30, 45, 60)

    def __init__(self, lanes: list[str], seed: int = 11):
        self.lanes = lanes
        self.random = random.Random(seed)
        self.q: defaultdict[tuple, dict[tuple[str, int], float]] = defaultdict(dict)

    @staticmethod
    def bucket(value: float) -> str:
        return "low" if value < 4 else "medium" if value < 10 else "high"

    def state(self, queues: dict[str, float]) -> tuple[str, ...]:
        return tuple(self.bucket(queues[lane]) for lane in self.lanes)

    def choose(self, queues: dict[str, float], explore: bool = False) -> tuple[str, int]:
        state = self.state(queues)
        actions = [(lane, seconds) for lane in self.lanes for seconds in self.ACTIONS]
        if explore and self.random.random() < 0.15:
            return self.random.choice(actions)
        values = {action: self.q[state].get(action, 0.0) for action in actions}
        best = max(values.values(), default=0.0)
        candidates = [action for action, value in values.items() if value == best]
        return max(candidates, key=lambda action: queues[action[0]])

    def train(self, episodes: int = 300) -> list[float]:
        rewards = []
        for _ in range(episodes):
            queues = {lane: self.random.uniform(1, 16) for lane in self.lanes}
            episode_reward = 0.0
            for _ in range(18):
                state = self.state(queues)
                action = self.choose(queues, explore=True)
                lane, seconds = action
                before = sum(queues.values())
                for current in queues:
                    arrivals = self.random.uniform(0.04, 0.14) * seconds
                    discharge = (0.11 if current == lane else 0.015) * seconds
                    queues[current] = max(0.0, queues[current] + arrivals - discharge)
                reward = -sum(queues.values()) - before * 0.05
                next_state = self.state(queues)
                future = max(self.q[next_state].values(), default=0.0)
                old = self.q[state].get(action, 0.0)
                self.q[state][action] = old + 0.15 * (reward + 0.92 * future - old)
                episode_reward += reward
            rewards.append(round(episode_reward, 2))
        return rewards
