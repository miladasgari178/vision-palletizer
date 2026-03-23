# Vision Palletizer: Coordinate Transformation & Robot Control

A backend service that coordinates a **Universal Robots (UR5e)** to pick items from a vision-defined location and place them into a **configurable grid pattern**.


---

## Table of Contents

- [Quick Start](#quick-start)
- [Technical Specifications](#technical-specifications)
- [Resources](#resources)

---

## Quick Start

### Prerequisites

- Docker Desktop running
- ~4GB free disk space (for URSim image)

> **Apple Silicon (M1/M2/M3) Note:** URSim is an x86 image that runs via emulation. You may see a platform warning — this is normal. The simulator will work but may be slower than on Intel machines.

### 1. Launch the Environment

Create a virtual environment:
```bash
python3.10 -m venv venv
```
Activate the virtual environment:
```bash
source venv/bin/activate
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
and at the end run the back end and URSim containers
```bash
docker-compose up -d
```

This starts:

| Service | Access |
|---------|--------|
| URSim (PolyScope UI) | http://localhost:6080/vnc.html |
| Backend API | http://localhost:8000/docs |

### 2. Power On the Robot

1. Open PolyScope at http://localhost:6080/vnc.html
2. Click the red button in the bottom-left corner
3. Click **ON** → **START** to enable the robot

### 3. Verify Connection

```bash
curl http://localhost:8000/health
```

---

## Technical Specifications

### Camera Mounting

| Parameter | Value |
|-----------|-------|
| Position (X, Y, Z) | 500mm, 300mm, 800mm |
| Orientation (Roll, Pitch, Yaw) | 15°, -10°, 45° |
| Rotation convention | Intrinsic rotations: Z → Y → X |
| Optical axis | Camera Z points toward the scene |

### Vision System

The "camera" in this exercise represents an upstream vision system that detects boxes and reports their positions in the **camera coordinate frame**.

A mock vision output is provided in `backend/data/camera_detections.json`:

```json
{
  "detections": [
    {"x_mm": 50.0, "y_mm": -30.0, "z_mm": 0.0, "yaw_deg": 0.0},
    {"x_mm": 120.0, "y_mm": 45.0, "z_mm": 0.0, "yaw_deg": 15.0},
    ...
  ]
}
```

**Your task:** Load detections from this file and transform each position from camera frame to robot base frame before commanding the robot to pick.

> The JSON structure can be replaced with any arbitrary data following the same format — your implementation should handle different detection sets.

### URSim Ports

| Port | Purpose |
|------|---------|
| 6080 | PolyScope UI (VNC) |
| 29999 | Dashboard |
| 30004 | RTDE |
| 30001 | Primary Interface |

---

## Resources

### Libraries

| Library | Purpose |
|---------|---------|
| [`ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/) | Real-time data exchange with UR robots |
| [`vention-state-machine`](https://pypi.org/project/vention-state-machine/) | State machine implementation |

---

## 🎥 Visualization & Assumptions

### Project Visualization

A visual demonstration of the system (robot motion, coordinate transformation, and palletizing behavior) can be found here:

👉 [Watch the demo](https://www.youtube.com/watch?v=u5MS-ReyJCU)

> The video shows UR5e pick-and-place into a grid.

👉 [Watch the demo](https://www.youtube.com/watch?v=7nrU5fDtYYI)

> The video shows UR5e pick-and-place into a grid while grasping the boxes with different orientations.

---

### Assumptions

The following assumptions were made in this implementation:

1. **Workspace Validation**  
   A simple workspace check is performed before executing robot motions. This is implemented as a bounded 3D box around the robot for simplicity. More advanced reachability checks (e.g., inverse kinematics validation via URScript) could be integrated for improved accuracy.

2. **Logging System**  
   A log file is initialized when the backend server starts. This file records all successfully placed box positions for traceability and debugging.

3. **Configurable Parameters**  
   Robot workspace limits (X, Y, Z) and motion safety parameters — including velocity, acceleration, `FALLBACK_DOWNWARD_ORIENTATION`, and `APPROACH_HEIGHT_OFFSET` — are configurable via `config.py`.

4. **Singularity Handling**  
   Singularities are not explicitly handled when executing Cartesian motions (`moveL`) through RTDE. Therefore, input box positions (from the vision system JSON) must be defined carefully relative to the robot base and home position to avoid unstable or undefined robot behavior.

---

