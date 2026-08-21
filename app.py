"""Streamlit control room for the AI Smart Traffic Management prototype."""

import pandas as pd
import streamlit as st

from traffic_ai.demo_data import LANES, make_demo_frame
from traffic_ai.evaluate import evaluate
from traffic_ai.signal_controller import RuleBasedController
from traffic_ai.tracking import StationaryTracker

st.set_page_config(page_title="Traffic Intelligence", layout="wide")

st.title("Traffic Intelligence Control Room")
st.caption("YOLOv8 vehicle intelligence, queue estimation, and adaptive intersection control")

with st.sidebar:
    st.header("Simulation controls")
    refresh = st.slider("Analysis frame", 0, 120, 42)
    north = st.slider("North approach", 0, 20, 9)
    east = st.slider("East approach", 0, 20, 5)
    south = st.slider("South approach", 0, 20, 3)
    west = st.slider("West approach", 0, 20, 7)
    counts = {"North": north, "East": east, "South": south, "West": west}
    st.divider()
    st.caption("Demo mode uses deterministic synthetic tracks. Upload an MP4 and enable YOLO mode once model weights are available.")

detections, lane_metrics = make_demo_frame(refresh, counts)
tracker = StationaryTracker()
tracker.update(detections)
metrics_df = pd.DataFrame([metric.__dict__ for metric in lane_metrics])
decision = RuleBasedController().decide(lane_metrics)

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Intersection snapshot")
    st.info("Live feed placeholder — YOLOv8n + ByteTrack pipeline ready for MP4 input")
    st.write("The demo feed is represented by the live lane metrics below. Connect an MP4 through `iter_video()` to replace this panel with annotated frames.")
with right:
    st.subheader("Signal state")
    st.metric("Active approach", decision.lane_id)
    st.metric("Green window", f"{decision.green_seconds}s")
    st.write(decision.reason)
    st.success("Controller online · rule-based adaptive")

st.subheader("Lane intelligence")
st.dataframe(
    metrics_df.rename(
        columns={
            "lane_id": "Approach",
            "vehicle_count": "Vehicles",
            "weighted_density": "Weighted density",
            "queue_length_m": "Queue (m)",
            "stationary_count": "Stationary",
            "occupancy_pct": "Occupancy %",
            "flow_veh_per_min": "Flow / min",
        }
    ).set_index("Approach"),
    use_container_width=True,
)
st.bar_chart(metrics_df.set_index("lane_id")[["weighted_density", "queue_length_m"]])

st.subheader("Controller benchmark")
st.caption("Demo processing target: 30 FPS · Detection mAP is shown as unavailable until an annotated validation set is supplied.")
if st.button("Run 15-minute benchmark", type="primary"):
    with st.spinner("Training Q-learning policy and simulating traffic flow..."):
        results, rewards = evaluate({metric.lane_id: metric.weighted_density for metric in lane_metrics})
    result_df = pd.DataFrame(results).set_index("Controller")
    st.dataframe(result_df, use_container_width=True)
    st.line_chart(pd.DataFrame({"Episode reward": rewards}))
    chart_df = result_df[["Average wait (s)", "Max queue (m)"]]
    st.bar_chart(chart_df)
else:
    st.caption("Compare fixed-time, rule-based, and Q-learning control with the benchmark button.")

with st.expander("Pipeline notes"):
    st.markdown(
        """
        **Detection:** `YOLODetector` filters COCO classes to car, bus, truck, motorcycle, and bicycle and uses Ultralytics ByteTrack IDs.

        **Estimation:** lane polygons assign detections to North, East, South, and West approaches. Density applies road-space weights; queue length measures the farthest stationary tracked vehicle from the stop line.

        **Control:** rule-based green time is `base_time + k × weighted_density`, capped between 15 and 60 seconds. Q-learning discretizes each lane as low / medium / high and learns 15 / 30 / 45 / 60 second actions.
        """
    )
