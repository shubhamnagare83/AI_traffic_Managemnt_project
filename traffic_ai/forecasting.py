"""2030 Predictive Traffic Horizon & Dynamic Tidal Lane Reversal."""

import random
from typing import Literal

WeatherType = Literal["Clear / Optima", "Cyber-Rain", "Dense Urban Fog", "Solar Flare / Sensor Noise"]


class TrafficForecaster:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def predict_horizon(
        self,
        current_counts: dict[str, int],
        weather: WeatherType = "Clear / Optima",
        horizon_minutes: int = 30,
    ) -> dict[str, list[dict]]:
        """Forecast demand trajectory for the next 30 minutes in 5-minute increments."""
        weather_factors = {
            "Clear / Optima": 1.0,
            "Cyber-Rain": 1.25,
            "Dense Urban Fog": 1.15,
            "Solar Flare / Sensor Noise": 1.10,
        }
        factor = weather_factors.get(weather, 1.0)

        intervals = list(range(0, horizon_minutes + 5, 5))
        series = []

        for minute in intervals:
            point = {"Minute": f"+{minute}m"}
            for lane, count in current_counts.items():
                # Temporal curve (simulated peak fluctuation + noise)
                wave = 1.0 + 0.35 * (minute / 15.0 - (minute / 30.0) ** 2)
                predicted = round(count * wave * factor + self.random.uniform(-1.5, 2.0))
                point[lane] = max(1, predicted)
            series.append(point)

        return series

    def evaluate_tidal_lane_reversal(self, lane_counts: dict[str, int]) -> dict[str, object]:
        """Assess whether directional asymmetry warrants dynamic reversible lane switching."""
        north_south = lane_counts.get("North", 0) + lane_counts.get("South", 0)
        east_west = lane_counts.get("East", 0) + lane_counts.get("West", 0)

        # Check for North-South asymmetry
        ns_diff = abs(lane_counts.get("North", 0) - lane_counts.get("South", 0))
        ew_diff = abs(lane_counts.get("East", 0) - lane_counts.get("West", 0))

        if ns_diff >= 8 and north_south > 12:
            heavy = "North" if lane_counts.get("North", 0) > lane_counts.get("South", 0) else "South"
            light = "South" if heavy == "North" else "North"
            return {
                "active": True,
                "corridor": "North-South Axis",
                "recommended_action": f"Reassign Lane 2 to {heavy}bound priority (Tidal flow active)",
                "asymmetry_ratio": round((lane_counts.get(heavy, 0) / max(north_south, 1)) * 100, 1),
                "delay_reduction_pct": 28.4,
            }

        if ew_diff >= 8 and east_west > 12:
            heavy = "East" if lane_counts.get("East", 0) > lane_counts.get("West", 0) else "West"
            return {
                "active": True,
                "corridor": "East-West Axis",
                "recommended_action": f"Reassign Lane 2 to {heavy}bound priority (Tidal flow active)",
                "asymmetry_ratio": round((lane_counts.get(heavy, 0) / max(east_west, 1)) * 100, 1),
                "delay_reduction_pct": 25.1,
            }

        return {
            "active": False,
            "corridor": "Balanced Multi-Approach",
            "recommended_action": "Standard 4-way balanced phase matrix",
            "asymmetry_ratio": 50.0,
            "delay_reduction_pct": 0.0,
        }
