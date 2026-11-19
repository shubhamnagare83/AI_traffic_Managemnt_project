"""Professional Computer Vision HUD, Vehicle Speed Radar, and Camera Pipeline.

Designed with clean, modern enterprise aesthetics: crisp typography, high-contrast speed pills,
anti-aliased lane geometry, and decluttered real-time overlays.
"""

from typing import Generator
import cv2
import numpy as np

from .density_estimator import LaneROI
from .models import Detection, LaneMetrics, SignalDecision, V2XTelemetry


# Professional, high-contrast enterprise color palette (BGR)
COLOR_PALETTE = {
    "speed_normal": (80, 210, 110),    # Emerald Green (Safe Speed)
    "speed_fast": (60, 70, 240),       # Coral Red (Speeding)
    "speed_slow": (30, 180, 245),      # Amber (Slow / Decelerating)
    "speed_idle": (140, 140, 150),     # Muted Gray (Stationary / Queued)
    "emergency": (50, 40, 245),        # Crimson Red (Emergency Priority)
    "lane_active": (80, 220, 120),     # Clean Green Line for Green Light Corridor
    "lane_hold": (70, 70, 200),        # Clean Red Line for Red Light Corridor
    "box_border": (230, 230, 230),     # Crisp White / Silver Bounding Box
    "pill_bg": (18, 24, 34),           # Deep Slate Background
}


class CyberHUDAnnotator:
    """Enterprise-grade, crystal-clear computer vision HUD with prominent vehicle speed indicators."""

    @classmethod
    def render_hud(
        cls,
        frame: np.ndarray,
        detections: list[Detection],
        lanes: list[LaneROI],
        signal_decision: SignalDecision | None = None,
        telemetry: V2XTelemetry | None = None,
        fps: float = 30.0,
        speed_limit_kmh: float = 50.0,
        model_name: str = "YOLOv8 + Speed Radar",
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # -------------------------------------------------------------
        # 1. Road Corridors & Stop Lines (Crisp & Decluttered)
        # -------------------------------------------------------------
        for lane in lanes:
            poly = np.array(lane.polygon, dtype=np.int32)
            if (w, h) != (800, 800):
                poly[:, 0] = (poly[:, 0] * w / 800).astype(np.int32)
                poly[:, 1] = (poly[:, 1] * h / 800).astype(np.int32)

            is_active = signal_decision and signal_decision.lane_id == lane.lane_id
            line_color = COLOR_PALETTE["lane_active"] if is_active else (50, 65, 85)
            line_thick = 2 if is_active else 1

            # Only draw boundary line — NO heavy polygon fill that obscures the road!
            cv2.polylines(out, [poly], isClosed=True, color=line_color, thickness=line_thick, lineType=cv2.LINE_AA)

            # Minimalist lane approach tag
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            status_tag = " [GO]" if is_active else " [STOP]"
            cv2.putText(
                out,
                f"{lane.lane_id.upper()}{status_tag}",
                (cx - 45, cy),
                cv2.FONT_HERSHEY_DUPLEX,
                0.40,
                line_color,
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # 2. Prominent Vehicle Bounding Boxes & SPEED BADGES
        # -------------------------------------------------------------
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if (w, h) != (800, 800):
                x1 = int(x1 * w / 800)
                x2 = int(x2 * w / 800)
                y1 = int(y1 * h / 800)
                y2 = int(y2 * h / 800)

            speed = det.speed_kmh if det.speed_kmh > 0 else (det.speed_px_frame * 3.2)
            is_emergency = det.is_emergency or det.vehicle_class == "emergency"

            # Speed status color selection
            if is_emergency:
                speed_color = COLOR_PALETTE["emergency"]
                badge_title = f"EMERGENCY #{det.track_id}"
                speed_str = f"{speed:.0f} km/h [PRIORITY]"
            elif speed <= 1.0:
                speed_color = COLOR_PALETTE["speed_idle"]
                badge_title = f"{det.vehicle_class.upper()} #{det.track_id}"
                speed_str = "0 km/h (QUEUED)"
            elif speed > speed_limit_kmh:
                speed_color = COLOR_PALETTE["speed_fast"]
                badge_title = f"{det.vehicle_class.upper()} #{det.track_id}"
                speed_str = f"{speed:.0f} km/h [SPEEDING]"
            elif speed < 20.0:
                speed_color = COLOR_PALETTE["speed_slow"]
                badge_title = f"{det.vehicle_class.upper()} #{det.track_id}"
                speed_str = f"{speed:.0f} km/h"
            else:
                speed_color = COLOR_PALETTE["speed_normal"]
                badge_title = f"{det.vehicle_class.upper()} #{det.track_id}"
                speed_str = f"{speed:.0f} km/h"

            # Draw clean bounding box with smooth corner accents
            cv2.rectangle(out, (x1, y1), (x2, y2), speed_color, 2, cv2.LINE_AA)

            # Draw prominent speed badge directly above the vehicle
            pill_w = 145
            pill_h = 32
            pill_x1 = max(4, min(w - pill_w - 4, x1))
            pill_y1 = max(4, y1 - pill_h - 4)
            pill_x2 = pill_x1 + pill_w
            pill_y2 = pill_y1 + pill_h

            # Pill Background with subtle border
            cv2.rectangle(out, (pill_x1, pill_y1), (pill_x2, pill_y2), COLOR_PALETTE["pill_bg"], -1)
            cv2.rectangle(out, (pill_x1, pill_y1), (pill_x2, pill_y2), speed_color, 1, cv2.LINE_AA)

            # Vehicle title (upper line)
            cv2.putText(
                out,
                badge_title,
                (pill_x1 + 6, pill_y1 + 13),
                cv2.FONT_HERSHEY_DUPLEX,
                0.34,
                (200, 210, 225),
                1,
                cv2.LINE_AA,
            )

            # Prominent speed readout (lower line - BOLD)
            cv2.putText(
                out,
                speed_str,
                (pill_x1 + 6, pill_y1 + 27),
                cv2.FONT_HERSHEY_DUPLEX,
                0.40,
                speed_color,
                1,
                cv2.LINE_AA,
            )

            # Subtle velocity indicator dot at vehicle center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(out, (cx, cy), 3, speed_color, -1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. Clean, Minimalist Header Status Ribbon
        # -------------------------------------------------------------
        header_h = 30
        cv2.rectangle(out, (0, 0), (w, header_h), (12, 16, 24), -1)
        cv2.line(out, (0, header_h), (w, header_h), (40, 50, 65), 1, cv2.LINE_AA)

        active_lane_name = signal_decision.lane_id.upper() if signal_decision else "NORTH"
        green_time = signal_decision.green_seconds if signal_decision else 30

        # Status text left
        cv2.putText(
            out,
            f"SIGNAL: {active_lane_name} [GREEN {green_time}s]",
            (12, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            COLOR_PALETTE["speed_normal"],
            1,
            cv2.LINE_AA,
        )

        # Speed limit indicator center
        cv2.putText(
            out,
            f"SPEED LIMIT: {speed_limit_kmh:.0f} km/h",
            (w // 2 - 70, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            (210, 220, 235),
            1,
            cv2.LINE_AA,
        )

        # Real-time FPS & detection count right
        cv2.putText(
            out,
            f"{len(detections)} VEHICLES | {fps:.1f} FPS",
            (w - 170, 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            (160, 175, 195),
            1,
            cv2.LINE_AA,
        )

        return out


class Synthetic2030Twin:
    """Procedurally renders a clean, professional intersection for live simulation."""

    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height

    def generate_frame(
        self,
        frame_index: int,
        detections: list[Detection],
        active_lane: str = "North",
        evp_active: bool = False,
    ) -> np.ndarray:
        w, h = self.width, self.height
        # Modern asphalt background (deep slate gray)
        frame = np.full((h, w, 3), (24, 28, 35), dtype=np.uint8)

        # 4-Way Road Surface (clean, smooth asphalt)
        road_surface_color = (36, 42, 52)
        cv2.rectangle(frame, (250, 0), (550, h), road_surface_color, -1)
        cv2.rectangle(frame, (0, 250), (w, 550), road_surface_color, -1)

        # Center intersection junction
        cv2.rectangle(frame, (250, 250), (550, 550), (42, 49, 60), -1)

        # Clean road curbs / boundary edges
        curb_color = (70, 80, 95)
        cv2.line(frame, (250, 0), (250, 250), curb_color, 2)
        cv2.line(frame, (550, 0), (550, 250), curb_color, 2)
        cv2.line(frame, (250, 550), (250, h), curb_color, 2)
        cv2.line(frame, (550, 550), (550, h), curb_color, 2)
        cv2.line(frame, (0, 250), (250, 250), curb_color, 2)
        cv2.line(frame, (0, 550), (250, 550), curb_color, 2)
        cv2.line(frame, (550, 250), (w, 250), curb_color, 2)
        cv2.line(frame, (550, 550), (w, 550), curb_color, 2)

        # White crosswalks
        cw_color = (85, 95, 110)
        for x in range(260, 540, 28):
            cv2.rectangle(frame, (x, 232), (x + 16, 246), cw_color, -1)
            cv2.rectangle(frame, (x, 554), (x + 16, 568), cw_color, -1)
        for y in range(260, 540, 28):
            cv2.rectangle(frame, (232, y), (246, y + 16), cw_color, -1)
            cv2.rectangle(frame, (554, y), (568, y + 16), cw_color, -1)

        # Dashed road center divider lines
        dash_len = 16
        gap = 14
        dash_color = (200, 200, 210)
        for y in range(0, 230, dash_len + gap):
            cv2.line(frame, (400, y), (400, y + dash_len), dash_color, 1)
        for y in range(570, h, dash_len + gap):
            cv2.line(frame, (400, y), (400, y + dash_len), dash_color, 1)
        for x in range(0, 230, dash_len + gap):
            cv2.line(frame, (x, 400), (x + dash_len, 400), dash_color, 1)
        for x in range(570, w, dash_len + gap):
            cv2.line(frame, (x, 400), (x + dash_len, 400), dash_color, 1)

        # Stop lines (Solid Green for active corridor, Solid Red for stopped corridors)
        stop_lines = {
            "North": ((250, 248), (400, 248)),
            "South": ((400, 552), (550, 552)),
            "West": ((248, 400), (248, 550)),
            "East": ((552, 250), (552, 400)),
        }
        for lane, (p1, p2) in stop_lines.items():
            is_green = (lane == active_lane)
            s_color = (60, 210, 100) if is_green else (50, 50, 220)
            cv2.line(frame, p1, p2, s_color, 3, cv2.LINE_AA)

        # Draw vehicle bodies on the road
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_emergency = det.is_emergency or det.vehicle_class == "emergency"
            vcolor = (50, 40, 245) if is_emergency else (55, 65, 80)

            # Sleek vehicle chassis
            cv2.rectangle(frame, (x1, y1), (x2, y2), vcolor, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 190, 205), 1)

            # Windshield
            w_inset = max(2, (x2 - x1) // 5)
            h_inset = max(2, (y2 - y1) // 4)
            cv2.rectangle(
                frame,
                (x1 + w_inset, y1 + h_inset),
                (x2 - w_inset, y2 - h_inset),
                (30, 36, 45),
                -1,
            )

        return frame


class CameraManager:
    """Manages OpenCV webcam devices and video stream reading."""

    @staticmethod
    def get_webcam_frame(camera_index: int = 0) -> tuple[bool, np.ndarray | None]:
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False, None
            ret, frame = cap.read()
            cap.release()
            return ret, frame
        except Exception:
            return False, None
