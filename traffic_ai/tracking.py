"""Small stateful tracker utilities used by density estimation."""

from dataclasses import dataclass

from .models import Detection


@dataclass
class TrackState:
    centroid: tuple[float, float]
    stationary_frames: int = 0


class StationaryTracker:
    def __init__(self, movement_threshold_px: float = 3.0, stationary_after_frames: int = 8):
        self.movement_threshold_px = movement_threshold_px
        self.stationary_after_frames = stationary_after_frames
        self.states: dict[int, TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            previous = self.states.get(detection.track_id)
            if previous:
                dx = detection.centroid[0] - previous.centroid[0]
                dy = detection.centroid[1] - previous.centroid[1]
                speed = (dx * dx + dy * dy) ** 0.5
                detection.speed_px_frame = speed
                if speed <= self.movement_threshold_px:
                    previous.stationary_frames += 1
                else:
                    previous.stationary_frames = 0
            else:
                detection.speed_px_frame = 0.0
                previous = TrackState(detection.centroid)
                self.states[detection.track_id] = previous
            detection.speed_px_frame = getattr(detection, "speed_px_frame", 0.0)
            if previous.stationary_frames >= self.stationary_after_frames:
                detection.speed_px_frame = 0.0
            previous.centroid = detection.centroid
        active_ids = {d.track_id for d in detections}
        self.states = {key: value for key, value in self.states.items() if key in active_ids}
        return detections
