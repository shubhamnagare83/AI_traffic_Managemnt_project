"""2030 Computer Vision Pipeline, Camera Capture, and Cyber-HUD Annotation."""

import math
import random
from typing import Generator
import cv2
import numpy as np

from .density_estimator import LaneROI
from .models import Detection, LaneMetrics, SignalDecision, V2XTelemetry


# BGR Colors for 2030 Cyberpunk HUD
COLORS = {
    "car": (254, 242, 0),          # Cyan
    "autonomous_pod": (136, 255, 0), # Emerald
    "ev_shuttle": (220, 200, 40),   # Cyan-Lime
    "truck": (0, 170, 255),        # Amber
    "bus": (0, 140, 255),          # Darker Amber
    "motorcycle": (255, 68, 192),  # Purple
    "bicycle": (255, 120, 180),    # Violet
    "emergency": (85, 34, 255),    # Crimson Red
    "hud_cyan": (255, 240, 0),
    "hud_green": (80, 255, 80),
    "hud_red": (70, 70, 255),
    "hud_amber": (0, 180, 255),
    "hud_dark": (15, 12, 10),
}


class CyberHUDAnnotator:
    """Overlays futuristic 2030 aerospace/cybernetic HUD graphics onto camera feeds."""

    @staticmethod
    def draw_corner_rect(
        img: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 10,
    ) -> None:
        """Draw a box with stylish cyberpunk corner brackets."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw semi-transparent bounding box fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        # Draw full thin box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Draw prominent corner brackets
        clen = min(corner_len, max(5, (x2 - x1) // 4), max(5, (y2 - y1) // 4))
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + clen, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + clen), color, thickness, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - clen, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + clen), color, thickness, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + clen, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - clen), color, thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - clen, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - clen), color, thickness, cv2.LINE_AA)

    @classmethod
    def render_hud(
        cls,
        frame: np.ndarray,
        detections: list[Detection],
        lanes: list[LaneROI],
        signal_decision: SignalDecision | None = None,
        telemetry: V2XTelemetry | None = None,
        fps: float = 30.0,
        model_name: str = "YOLOv8n + Deep-MARL 2030",
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw Lane Polygons & Occupancy Heatmaps
        overlay = out.copy()
        for lane in lanes:
            poly = np.array(lane.polygon, dtype=np.int32)
            # Scale if frame is different from 800x800 demo
            if (w, h) != (800, 800):
                poly[:, 0] = (poly[:, 0] * w / 800).astype(np.int32)
                poly[:, 1] = (poly[:, 1] * h / 800).astype(np.int32)

            is_active = signal_decision and signal_decision.lane_id == lane.lane_id
            poly_color = (0, 220, 100) if is_active else (220, 100, 0)
            cv2.polylines(out, [poly], isClosed=True, color=poly_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.fillPoly(overlay, [poly], poly_color)

            # Label approach
            cx = int(np.mean(poly[:, 0]))
            cy = int(np.mean(poly[:, 1]))
            cv2.putText(
                out,
                f"[{lane.lane_id.upper()} CORRIDOR]",
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

        # 2. Draw Vehicle Detections with Cyberpunk Brackets & Telemetry Tags
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            if (w, h) != (800, 800):
                x1 = int(x1 * w / 800)
                x2 = int(x2 * w / 800)
                y1 = int(y1 * h / 800)
                y2 = int(y2 * h / 800)

            vclass = det.vehicle_class
            color = COLORS.get(vclass, (0, 255, 255))
            if det.is_emergency:
                color = COLORS["emergency"]

            cls.draw_corner_rect(out, (x1, y1), (x2, y2), color, thickness=2)

            # Draw target reticle at centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

            # Draw speed velocity arrow
            if det.speed_px_frame > 0.5:
                speed_kmh = det.speed_kmh or round(det.speed_px_frame * 3.2, 1)
                cv2.line(out, (cx, cy), (cx, cy - int(det.speed_px_frame * 3)), color, 1, cv2.LINE_AA)

            # HUD Label badge
            tag_text = f"#{det.track_id:02d} {vclass.upper()}"
            if det.is_cav:
                tag_text += " [L4-CAV]"
            if det.is_emergency:
                tag_text = f"🚨 EVP: #{det.track_id} EMERGENCY"

            sub_text = f"{det.speed_kmh:.0f}km/h | GLOSA:{det.glosa_speed_kmh:.0f}"

            # Badge background
            cv2.rectangle(out, (x1, max(0, y1 - 26)), (x1 + 140, y1), (20, 20, 24), -1)
            cv2.rectangle(out, (x1, max(0, y1 - 26)), (x1 + 140, y1), color, 1)

            cv2.putText(
                out,
                tag_text,
                (x1 + 4, max(10, y1 - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                sub_text,
                (x1 + 4, max(22, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # 3. Top Cybernetic Telemetry Banner
        banner_h = 36
        banner_overlay = out.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (10, 14, 22), -1)
        cv2.addWeighted(banner_overlay, 0.85, out, 0.15, 0, out)
        cv2.line(out, (0, banner_h), (w, banner_h), (0, 242, 254), 1, cv2.LINE_AA)

        # Top banner text items
        cv2.putText(
            out,
            f"SYSTEM: 2030 QUANTUM-EDGE AI",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 242, 254),
            1,
            cv2.LINE_AA,
        )

        active_str = f"ACTIVE PHASE: {signal_decision.lane_id.upper() if signal_decision else 'NORTH'}"
        phase_color = (0, 255, 120) if (not signal_decision or not signal_decision.evp_preemption) else (85, 34, 255)
        cv2.putText(
            out,
            active_str,
            (280, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            phase_color,
            1,
            cv2.LINE_AA,
        )

        if telemetry and telemetry.evp_active:
            cv2.putText(
                out,
                f"🚨 EVP ACTIVE: {telemetry.evp_lane}",
                (480, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (85, 34, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            v2x_str = f"V2X: {telemetry.connected_ratio_pct if telemetry else 82.4}% CAV"
            cv2.putText(
                out,
                v2x_str,
                (480, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 255, 100),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            out,
            f"FPS: {fps:.1f}",
            (w - 90, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return out


class Synthetic2030Twin:
    """Procedurally renders a 2030 Smart Intersection frame for live simulation."""

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
        # Dark high-tech cyber asphalt background
        frame = np.full((h, w, 3), (18, 22, 28), dtype=np.uint8)

        # Draw Road Network (4-Way Intersection)
        # Vertical road: x in [250, 550]
        # Horizontal road: y in [250, 550]
        road_color = (32, 38, 48)
        cv2.rectangle(frame, (250, 0), (550, h), road_color, -1)
        cv2.rectangle(frame, (0, 250), (w, 550), road_color, -1)

        # Center intersection box
        cv2.rectangle(frame, (250, 250), (550, 550), (40, 48, 60), -1)

        # Crosswalks with modern neon lines
        crosswalk_color = (90, 110, 130)
        # North crosswalk
        for x in range(260, 540, 30):
            cv2.rectangle(frame, (x, 230), (x + 18, 245), crosswalk_color, -1)
        # South crosswalk
        for x in range(260, 540, 30):
            cv2.rectangle(frame, (x, 555), (x + 18, 570), crosswalk_color, -1)
        # West crosswalk
        for y in range(260, 540, 30):
            cv2.rectangle(frame, (230, y), (245, y + 18), crosswalk_color, -1)
        # East crosswalk
        for y in range(260, 540, 30):
            cv2.rectangle(frame, (555, y), (570, y + 18), crosswalk_color, -1)

        # Dashed yellow & cyan cyber lane dividers
        dash_len = 16
        gap = 14
        # North-South center divider
        for y in range(0, 230, dash_len + gap):
            cv2.line(frame, (400, y), (400, y + dash_len), (0, 220, 255), 2)
        for y in range(570, h, dash_len + gap):
            cv2.line(frame, (400, y), (400, y + dash_len), (0, 220, 255), 2)
        # East-West center divider
        for x in range(0, 230, dash_len + gap):
            cv2.line(frame, (x, 400), (x + dash_len, 400), (0, 220, 255), 2)
        for x in range(570, w, dash_len + gap):
            cv2.line(frame, (x, 400), (x + dash_len, 400), (0, 220, 255), 2)

        # Stop Lines (Glowing Green for active lane, Red for stopped lanes)
        stop_lines = {
            "North": ((250, 248), (400, 248)),
            "South": ((400, 552), (550, 552)),
            "West": ((248, 400), (248, 550)),
            "East": ((552, 250), (552, 400)),
        }
        for lane, (p1, p2) in stop_lines.items():
            is_green = (lane == active_lane)
            s_color = (0, 255, 120) if is_green else (50, 50, 255)
            cv2.line(frame, p1, p2, s_color, 4, cv2.LINE_AA)

        # Draw vehicle bodies onto the road
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            vcolor = COLORS.get(det.vehicle_class, (0, 255, 255))
            if det.is_emergency:
                vcolor = (85, 34, 255)

            # Vehicle chassis
            cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 30, 38), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), vcolor, 2)

            # Windshield / glass
            w_inset = max(2, (x2 - x1) // 5)
            h_inset = max(2, (y2 - y1) // 4)
            cv2.rectangle(
                frame,
                (x1 + w_inset, y1 + h_inset),
                (x2 - w_inset, y2 - h_inset),
                (60, 80, 100),
                -1,
            )

            # LED Headlights
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (x1 + 3, y1 + 3), 2, (255, 255, 255), -1)
            cv2.circle(frame, (x2 - 3, y1 + 3), 2, (255, 255, 255), -1)

        return frame


class CameraManager:
    """Manages OpenCV webcam devices and video stream reading."""

    @staticmethod
    def get_webcam_frame(camera_index: int = 0) -> tuple[bool, np.ndarray | None]:
        """Attempt to grab a single frame from the specified webcam index."""
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False, None
            ret, frame = cap.read()
            cap.release()
            return ret, frame
        except Exception:
            return False, None
