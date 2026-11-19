# 🚦 AI Smart Traffic Management & Vehicle Speed Radar

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange.svg)](https://ultralytics.com/)
[![Tests](https://img.shields.io/badge/Tests-5%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

A professional, enterprise-grade Intelligent Transportation System (ITS) developed for final-year engineering capstone evaluation. Integrates edge computer vision for **real-time vehicle speed estimation**, **Webster's minimum delay signal optimization**, **automated law enforcement (e-Challan)**, and **emergency vehicle preemption (EVP)**.

---

## ⚡ Core Capabilities

### 1. ⚡ Real-Time Vehicle Speed Radar (Hero Feature)
- **Euclidean Displacement Calculation:** Real-time speed estimation calibrated from spatial pixel displacement:
  $$v_{\text{km/h}} = \left( \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\kappa} \times f \right) \times 3.6$$
- **High-Contrast Speed Badges:** Every vehicle displays a prominent, high-legibility speed pill (e.g. `48 km/h`, `0 km/h (QUEUED)`, `62 km/h [SPEEDING]`).
- **Color-Coded Telemetry:**
  - 🟢 **Emerald Green:** Normal cruising speed ($20 - 50\text{ km/h}$)
  - 🔴 **Coral Red:** Over-speeding violation ($> 50\text{ km/h}$)
  - 🟡 **Amber:** Slow moving / decelerating ($1 - 19\text{ km/h}$)
  - ⚪ **Muted Slate:** Queued at red signal ($0\text{ km/h}$)

### 2. 🛣️ "Fix This Road" — Road Geometry Presets
- **4-Way Urban Cross Intersection:** 4 independent corridors with dedicated stop lines and lane polygon boundaries.
- **T-Junction (3-Way):** 3-approach intersection.
- **Dual-Carriageway Arterial Road:** Inbound vs Outbound continuous flow with speed radar traps.

### 3. 📐 Webster's Optimum Delay Formulation (Civil Engineering Standard)
Implements F.V. Webster's (1958) classic mathematical model to eliminate unnecessary intersection delay:
$$C_0 = \frac{1.5 L + 5}{1 - Y}, \quad g_i = \frac{y_i}{Y} (C_0 - L)$$

### 4. 🚨 Automated Traffic Law Enforcement (e-Challan)
- **Red-Light Runner Detection:** Flags vehicles crossing the stop line during active red signals.
- **Speed Enforcement Radar:** Generates digital citation records with recorded speed, timestamp, and fine amounts in ₹.

### 5. 🚑 Emergency Vehicle Preemption (EVP)
- Instant 1-click or automated green-wave clearance for inbound ambulances, halting conflicting cross-traffic.

---

## 🖥️ Clean Enterprise Interface

The UI is designed around a modern, uncluttered enterprise architecture:
- **Top Metrics Row:** Active Phase timer, Average Traffic Speed, Tracked Vehicle Count, and Violation Counter.
- **Left Panel (60%):** Large High-Definition Video Stream with speed badges and camera mode switcher (Digital Twin, Live Hardware Webcam, Snapshot, Video Upload).
- **Right Panel (40%):** Real-time vehicle speed table, approach queue density, and emergency dispatch controls.
- **Lower Section:** Organized tabs for e-Challan records, Webster signal optimization, predictive forecasting, and academic documentation.

---

## 🚀 Quickstart & Execution

### 1. Launch with 1-Click (Windows)
Double-click `run_dashboard.bat` or run:
```powershell
.\run_dashboard.bat
```

### 2. Manual Execution
```powershell
# Activate dedicated virtual environment
.\.venv\Scripts\Activate.ps1

# Run Streamlit on localhost:5000
streamlit run app.py --server.port 5000
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

### 3. Run Automated Unit Tests
```powershell
pytest -v
# 5 passed in 0.26s
```

---

## 🎓 Academic Dossier
Refer to [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the complete academic capstone report including problem statement, mathematical proofs, experimental results, and comparative benchmarks.
