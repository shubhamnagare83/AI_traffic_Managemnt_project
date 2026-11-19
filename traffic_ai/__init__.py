"""AI Smart Traffic Management - 2030 Quantum Edge Package."""

from .models import Detection, LaneMetrics, SignalDecision, IncidentAlert, V2XTelemetry, EcoMetrics
from .density_estimator import DensityEstimator, LaneROI
from .signal_controller import FixedTimeController, RuleBasedController, WebsterController
from .rl_agent import QLearningAgent
from .deep_marl_controller import DeepMARLController
from .v2x_system import V2XSystem
from .anomaly_detector import AnomalyDetector
from .violation_detector import ViolationDetector, TrafficViolation
from .eco_tracker import EcoTracker
from .forecasting import TrafficForecaster
from .camera_vision import CyberHUDAnnotator, CameraManager, Synthetic2030Twin
from .detection import YOLODetector, iter_video
from .demo_data import LANES, ROAD_PRESETS, make_demo_frame
