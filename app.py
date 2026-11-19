"""AI-Based Smart Traffic Management System.

Final Year Engineering Capstone Project.
Features:
- Real-Time Computer Vision Vehicle Speed Radar (YOLOv8 + ByteTrack)
- Prominent Vehicle Speed Badges (km/h) & Trajectory Tracking
- Configurable Road Geometry (4-Way Intersection, T-Junction, Dual-Carriageway)
- Webster's Minimum Delay Formulation & Deep-MARL Signal Optimization
- Automated Law Enforcement (Red Light Runner & Over-Speeding e-Challan)
- Emergency Vehicle Preemption (EVP) Green Wave
"""

import os
import tempfile
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traffic_ai.anomaly_detector import AnomalyDetector
from traffic_ai.camera_vision import CameraManager, CyberHUDAnnotator, Synthetic2030Twin
from traffic_ai.deep_marl_controller import DeepMARLController
from traffic_ai.demo_data import LANES, ROAD_PRESETS, make_demo_frame
from traffic_ai.density_estimator import DensityEstimator
from traffic_ai.detection import YOLODetector
from traffic_ai.eco_tracker import EcoTracker
from traffic_ai.evaluate import evaluate
from traffic_ai.forecasting import TrafficForecaster
from traffic_ai.models import Detection, SignalDecision
from traffic_ai.signal_controller import FixedTimeController, RuleBasedController, WebsterController
from traffic_ai.tracking import StationaryTracker
from traffic_ai.v2x_system import V2XSystem
from traffic_ai.violation_detector import ViolationDetector

# ---------------------------------------------------------
# Page Configuration & Clean Enterprise Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Smart Traffic Management & Speed Radar",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean Enterprise Dark Theme */
    .main {
        background-color: #090d16;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 95%;
    }
    
    /* Header Typography */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.90rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-badge-evp {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        animation: pulse 1.5s infinite;
    }

    /* Metric Card Styling */
    .metric-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.80rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-value-green {
        color: #10b981;
    }
    .metric-value-cyan {
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Clean Section Containers */
    .feed-container {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "v2x_system" not in st.session_state:
    st.session_state.v2x_system = V2XSystem()
if "anomaly_detector" not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()
if "violation_detector" not in st.session_state:
    st.session_state.violation_detector = ViolationDetector(speed_limit_kmh=50.0)
if "eco_tracker" not in st.session_state:
    st.session_state.eco_tracker = EcoTracker()
if "forecaster" not in st.session_state:
    st.session_state.forecaster = TrafficForecaster()
if "twin" not in st.session_state:
    st.session_state.twin = Synthetic2030Twin()
if "yolo_detector" not in st.session_state:
    st.session_state.yolo_detector = None

# ---------------------------------------------------------
# Sidebar: Clean Configuration Panel
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    st.markdown("**Road Geometry Layout**")
    selected_preset_name = st.selectbox(
        "Select Road Intersection",
        list(ROAD_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    active_lanes = ROAD_PRESETS[selected_preset_name]

    st.markdown("**Signal Control Algorithm**")
    controller_choice = st.selectbox(
        "Active Controller",
        [
            "Webster's Optimal Delay (Civil Standard)",
            "Deep-MARL Multi-Agent AI",
            "Rule-Based Adaptive",
            "Fixed-Time (Legacy)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Speed Limit & Computer Vision**")
    speed_limit = st.slider("Speed Limit (km/h)", 30, 80, 50, 5)
    st.session_state.violation_detector.speed_limit_kmh = float(speed_limit)

    conf_thresh = st.slider("Detection Confidence", 0.15, 0.85, 0.35, 0.05)

    st.divider()
    st.markdown("**Traffic Demand per Approach**")
    lane_keys = [l.lane_id for l in active_lanes]
    counts = {}
    default_vals = {"North": 8, "East": 5, "South": 3, "West": 6, "Inbound Northbound": 10, "Outbound Southbound": 5}
    for lk in lane_keys:
        def_v = default_vals.get(lk, 5)
        counts[lk] = st.slider(f"{lk} approach", 0, 20, def_v)

    st.divider()
    st.markdown("**Emergency Vehicle Preemption**")
    evp_lane = st.selectbox("Corridor Target", lane_keys, label_visibility="collapsed")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚨 Dispatch", use_container_width=True, type="primary"):
            st.session_state.v2x_system.trigger_emergency_preemption(evp_lane)
            st.toast(f"Emergency Preemption Active on {evp_lane}!", icon="🚨")
    with col_btn2:
        if st.button("Reset Phase", use_container_width=True):
            st.session_state.v2x_system.cancel_emergency_preemption()
            st.toast("Normal signal matrix restored.", icon="✅")

# ---------------------------------------------------------
# Core Controller Execution & Data Processing
# ---------------------------------------------------------
def get_yolo(conf: float = 0.35):
    if st.session_state.yolo_detector is None or st.session_state.yolo_detector.confidence != conf:
        with st.spinner("Initializing YOLOv8 Neural Engine..."):
            st.session_state.yolo_detector = YOLODetector("yolov8n.pt", confidence=conf)
    return st.session_state.yolo_detector


# Generate detections matching road layout
detections, lane_metrics = make_demo_frame(0, counts, lanes_preset=active_lanes)

# Track stationary states
tracker = StationaryTracker()
tracker.update(detections)

# Inject emergency vehicle if EVP is triggered
if st.session_state.v2x_system.evp_active:
    evp_target = st.session_state.v2x_system.evp_lane
    emergency_det = Detection(
        track_id=999,
        vehicle_class="emergency",
        confidence=0.99,
        bbox=(380, 260, 420, 310) if "North" in evp_target or "South" in evp_target else (260, 380, 310, 420),
        centroid=(400.0, 285.0),
        lane_id=evp_target,
        speed_px_frame=22.5,
        speed_kmh=72.0,
        is_cav=True,
        is_emergency=True,
        v2x_id="EMERGENCY-EVP-01",
    )
    detections.append(emergency_det)

# Update density metrics
estimator = DensityEstimator(active_lanes)
lane_metrics = estimator.estimate(detections)

# Compute Signal Decision
if "Webster" in controller_choice:
    controller = WebsterController()
    decision = controller.decide(lane_metrics)
elif "Deep-MARL" in controller_choice:
    controller = DeepMARLController()
    decision = controller.decide(
        lane_metrics,
        evp_active=st.session_state.v2x_system.evp_active,
        evp_lane=st.session_state.v2x_system.evp_lane,
    )
elif "Rule-Based" in controller_choice:
    controller = RuleBasedController()
    decision = controller.decide(lane_metrics)
else:
    queues = {m.lane_id: m.weighted_density for m in lane_metrics}
    controller = FixedTimeController(green_seconds=30)
    chosen_lane, green_sec = controller.choose(queues)
    decision = SignalDecision(chosen_lane, green_sec, "Fixed 30s cycle", "Fixed-Time")

# Update vehicle speeds & V2X telemetry
detections, telemetry = st.session_state.v2x_system.process_detections(
    detections,
    active_lane=decision.lane_id,
    remaining_green_sec=decision.green_seconds,
)

# Detect violations (Red light runners & speeding)
violations = st.session_state.violation_detector.check_violations(
    detections,
    active_lane=decision.lane_id,
    signal_state="GREEN",
    speed_limit_kmh=float(speed_limit),
)

# Anomaly scan & Eco computation
incidents = st.session_state.anomaly_detector.scan(
    detections,
    lane_metrics,
    active_lane=decision.lane_id,
    evp_active=st.session_state.v2x_system.evp_active,
)
eco = st.session_state.eco_tracker.compute(lane_metrics, controller_type=controller.name)

# Calculate summary speed statistics
speeds = [d.speed_kmh for d in detections if d.speed_kmh > 0]
avg_speed = round(float(np.mean(speeds)), 1) if speeds else 0.0
max_speed = round(float(np.max(speeds)), 1) if speeds else 0.0
moving_count = len(speeds)
queued_count = len(detections) - moving_count

# ---------------------------------------------------------
# Top Header Bar
# ---------------------------------------------------------
evp_active = st.session_state.v2x_system.evp_active
status_badge_html = (
    f'<span class="status-badge-evp">🚨 EMERGENCY PREEMPTION: {st.session_state.v2x_system.evp_lane}</span>'
    if evp_active
    else '<span class="status-badge">🟢 System Online · 30 FPS · Computer Vision Active</span>'
)

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="app-title">AI Smart Traffic Management & Vehicle Speed Radar</div>
            <div class="app-subtitle">Computer Vision Vehicle Speed Tracking · Webster Optimization · Intelligent Law Enforcement</div>
        </div>
        <div>
            {status_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Top 4 Clean Metric Cards
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Active Signal Phase</div>
            <div class="metric-value metric-value-green">{decision.lane_id.upper()}</div>
            <div class="metric-sub">Green Window: <strong>{decision.green_seconds} seconds</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Vehicle Speed</div>
            <div class="metric-value metric-value-cyan">{avg_speed} km/h</div>
            <div class="metric-sub">Speed Limit: {speed_limit} km/h · Peak: {max_speed} km/h</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Tracked Vehicles</div>
            <div class="metric-value">{len(detections)}</div>
            <div class="metric-sub">Moving: <strong>{moving_count}</strong> · Queued: <strong>{queued_count}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    viol_count = len(violations)
    viol_color = "style='color:#ef4444;'" if viol_count > 0 else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Traffic Violations</div>
            <div class="metric-value" {viol_color}>{viol_count}</div>
            <div class="metric-sub">Total e-Challan: <strong>₹{viol_count * 1000:,}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Main Workspace: 2 Clean Balanced Columns
# ---------------------------------------------------------
col_vision, col_radar = st.columns([1.5, 1.0])

with col_vision:
    st.markdown("#### 📹 Computer Vision Feed & Speed Radar")
    
    # Clean inline source selection pills
    feed_mode = st.radio(
        "Camera Mode",
        ["🌐 Digital Twin Simulation", "📹 Live Hardware Webcam", "📸 Browser Snapshot", "📁 Upload Video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if feed_mode == "🌐 Digital Twin Simulation":
        twin_frame = st.session_state.twin.generate_frame(
            frame_index=0,
            detections=detections,
            active_lane=decision.lane_id,
            evp_active=evp_active,
        )
        annotated = CyberHUDAnnotator.render_hud(
            twin_frame,
            detections=detections,
            lanes=active_lanes,
            signal_decision=decision,
            telemetry=telemetry,
            fps=30.0,
            speed_limit_kmh=float(speed_limit),
        )
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"Road Layout: {selected_preset_name} · Real-time vehicle speed (km/h) badge overlay",
        )

    elif feed_mode == "📹 Live Hardware Webcam":
        cam_c1, cam_c2 = st.columns([1.5, 1])
        with cam_c1:
            cam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=4, value=0)
        with cam_c2:
            run_cam = st.checkbox("🔴 Start Live Stream", value=False)

        video_placeholder = st.empty()
        status_box = st.empty()

        if run_cam:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                st.error(f"Cannot access camera device {cam_idx}. Verify device connection.")
            else:
                detector = get_yolo(conf_thresh)
                frame_count = 0
                t_start = time.time()

                while run_cam and frame_count < 60:
                    ret, raw_frame = cap.read()
                    if not ret:
                        status_box.warning("Unable to read camera frame.")
                        break

                    frame_count += 1
                    cam_dets = detections
                    if frame_count % 2 == 0 and detector is not None:
                        detected = detector.track(raw_frame, frame_count)
                        if detected:
                            cam_dets = detected

                    fps_val = frame_count / max(time.time() - t_start, 0.001)

                    annotated_cam = CyberHUDAnnotator.render_hud(
                        raw_frame,
                        detections=cam_dets,
                        lanes=active_lanes,
                        signal_decision=decision,
                        telemetry=telemetry,
                        fps=fps_val,
                        speed_limit_kmh=float(speed_limit),
                    )
                    video_placeholder.image(
                        cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption=f"Hardware Camera Live Stream · {fps_val:.1f} FPS",
                    )
                    time.sleep(0.03)

                cap.release()
                status_box.info("Stream loop paused. Toggle checkbox to resume live stream.")
        else:
            st.info("Check 'Start Live Stream' above to stream directly from your physical webcam.")

    elif feed_mode == "📸 Browser Snapshot":
        camera_img = st.camera_input("Capture live snapshot from browser camera")
        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            detector = get_yolo(conf_thresh)
            snap_detections = detector.track(cv_img, 0)

            annotated_snap = CyberHUDAnnotator.render_hud(
                cv_img,
                detections=snap_detections if snap_detections else detections,
                lanes=active_lanes,
                signal_decision=decision,
                telemetry=telemetry,
                speed_limit_kmh=float(speed_limit),
            )
            st.image(
                cv2.cvtColor(annotated_snap, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Snapshot Analysis · {len(snap_detections)} Objects Detected",
            )

    elif feed_mode == "📁 Upload Video":
        uploaded_video = st.file_uploader("Upload Traffic Video (.mp4 / .avi / .mov)", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            ret, first_frame = cap.read()
            cap.release()

            if ret:
                detector = get_yolo(conf_thresh)
                vid_dets = detector.track(first_frame, 0)
                annotated_vid = CyberHUDAnnotator.render_hud(
                    first_frame,
                    detections=vid_dets,
                    lanes=active_lanes,
                    signal_decision=decision,
                    telemetry=telemetry,
                    speed_limit_kmh=float(speed_limit),
                )
                st.image(
                    cv2.cvtColor(annotated_vid, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=f"Uploaded Video Frame Analysis · {len(vid_dets)} Vehicles Detected",
                )
            try:
                os.unlink(tfile.name)
            except Exception:
                pass

with col_radar:
    st.markdown("#### ⚡ Live Vehicle Speed & Telemetry")
    
    # Real-time Vehicle Speed Table
    vehicle_rows = []
    for d in detections:
        spd = d.speed_kmh
        if d.is_emergency:
            status_tag = "🚨 EMERGENCY"
        elif spd == 0:
            status_tag = "QUEUED"
        elif spd > speed_limit:
            status_tag = "⚠️ SPEEDING"
        else:
            status_tag = "NORMAL"

        vehicle_rows.append(
            {
                "Track ID": f"#{d.track_id:02d}",
                "Vehicle Class": d.vehicle_class.upper(),
                "Speed": f"{spd:.1f} km/h",
                "Approach": d.lane_id or "Active",
                "Status": status_tag,
            }
        )

    v_df = pd.DataFrame(vehicle_rows)
    st.dataframe(v_df, use_container_width=True, height=280)

    # Approach Lane Metrics Table
    st.markdown("#### 📊 Approach Density & Queues")
    lane_table = pd.DataFrame(
        [
            {
                "Approach": m.lane_id,
                "Vehicles": m.vehicle_count,
                "Queue (m)": f"{m.queue_length_m:.1f}m",
                "Flow/min": m.flow_veh_per_min,
                "Occupancy": f"{m.occupancy_pct}%",
            }
            for m in lane_metrics
        ]
    ).set_index("Approach")
    st.dataframe(lane_table, use_container_width=True, height=180)

# ---------------------------------------------------------
# Secondary Tabs: Decluttered & Professional
# ---------------------------------------------------------
st.divider()

t1, t2, t3, t4, t5 = st.tabs(
    [
        "⚖️ Law Enforcement (e-Challan)",
        "🚦 Signal Matrix & Optimization",
        "🔮 Predictive Forecasting & Benchmark",
        "🌿 Carbon & Eco-Footprint",
        "🎓 Capstone Project Dossier",
    ]
)

# TAB 1: Law Enforcement & Violations
with t1:
    st.markdown("### 🚨 Intelligent Law Enforcement & Violation Radar (e-Challan)")
    st.caption("Automated computer vision detection of red light runners and speeding offenders at intersection stop lines.")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.metric("Total Violations Flagged", len(violations))
    with v2:
        st.metric("Active Speed Limit", f"{speed_limit} km/h")
    with v3:
        total_penalties = sum(v.fine_amount_inr for v in violations)
        st.metric("Total Penalties Generated", f"₹{total_penalties:,}")

    if violations:
        v_report = pd.DataFrame(
            [
                {
                    "Citation ID": v.violation_id,
                    "Timestamp": v.timestamp,
                    "Vehicle Type": v.vehicle_class,
                    "Approach": v.lane_id,
                    "Violation Type": v.violation_type.replace("_", " "),
                    "Recorded Speed": f"{v.recorded_speed_kmh} km/h",
                    "Signal State": v.signal_phase,
                    "Penalty Amount": f"₹{v.fine_amount_inr}",
                    "Status": v.evidence_status,
                }
                for v in reversed(violations[-8:])
            ]
        ).set_index("Citation ID")
        st.dataframe(v_report, use_container_width=True)
    else:
        st.success("No active violations detected. 100% traffic compliance across all corridors.")

# TAB 2: Signal Matrix & Optimization
with t2:
    st.markdown("### 🚦 Signal Phase Allocation & Webster Optimization")
    st.info(f"**Controller Reasoning:** {decision.reason}")

    c_phases = [l.lane_id for l in active_lanes]
    c_colors = ["#10b981" if p == decision.lane_id else "#ef4444" for p in c_phases]
    c_status = ["GREEN (FLOW)" if p == decision.lane_id else "RED (HOLD)" for p in c_phases]
    c_times = [decision.green_seconds if p == decision.lane_id else 45 for p in c_phases]

    fig_phase = go.Figure(
        data=[
            go.Bar(
                x=c_phases,
                y=c_times,
                marker_color=c_colors,
                text=c_status,
                textposition="auto",
            )
        ]
    )
    fig_phase.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis_title="Phase Allocation (seconds)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# TAB 3: Predictive Forecasting & Benchmark
with t3:
    st.markdown("### 🔮 30-Minute Demand Forecast & Multi-Controller Benchmark")
    
    forecast_data = st.session_state.forecaster.predict_horizon(counts, horizon_minutes=30)
    f_df = pd.DataFrame(forecast_data).set_index("Minute")

    fig_line = px.line(
        f_df,
        markers=True,
        title="30-Minute Projected Vehicle Arrival Curve",
    )
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    if st.button("🚀 Run Multi-Controller Benchmark Simulation (900s)", type="primary"):
        with st.spinner("Simulating traffic flow across 5 controller paradigms..."):
            results, _ = evaluate({m.lane_id: m.weighted_density for m in lane_metrics})
        b_df = pd.DataFrame(results).set_index("Controller")
        st.dataframe(b_df, use_container_width=True)

# TAB 4: Carbon & Eco-Footprint
with t4:
    st.markdown("### 🌿 Urban Eco-Footprint & Emissions Telemetry")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("CO2 Avoided", f"{eco.co2_saved_kg} kg", delta=f"+{eco.co2_saved_kg:.1f} kg")
    with e2:
        st.metric("Fuel Conserved", f"{eco.fuel_saved_liters} Liters", delta=f"+{eco.fuel_saved_liters:.1f} L")
    with e3:
        st.metric("EV Energy Saved", f"{eco.ev_energy_saved_kwh} kWh", delta=f"+{eco.ev_energy_saved_kwh:.1f} kWh")
    with e4:
        st.metric("Noise Dampening", f"-{eco.noise_reduction_db} dB", delta=f"-{eco.noise_reduction_db} dB")

# TAB 5: Capstone Dossier
with t5:
    st.markdown("### 🎓 Capstone Academic Dossier & Formulations")
    st.markdown(
        r"""
        #### 1. Webster's Minimum Delay Formulation (1958)
        $$C_0 = \frac{1.5 L + 5}{1 - Y}$$
        - $L = \sum l_i$: Total lost time per cycle ($\approx 16\text{s}$ for 4 phases).
        - $Y = \sum \frac{q_i}{s_i}$: Sum of approach arrival flows divided by saturation flow ($s_i = 1800\text{ veh/hr}$).
        - Effective green time: $g_i = \frac{y_i}{Y} (C_0 - L)$.

        #### 2. Speed Formulation via Euclidean Displacement
        $$v_{\text{km/h}} = \left( \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{\kappa} \times f \right) \times 3.6$$
        where $\kappa = 12.0\text{ pixels/meter}$ and $f = 30\text{ fps}$.
        """
    )
