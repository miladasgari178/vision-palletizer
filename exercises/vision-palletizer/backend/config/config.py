"""Configuration file for the backend of the application."""

import numpy as np

# camera detected box position path
DETECTED_BOX_POSITION_PATH = "./data/camera_detections.json"

# Robot pose logging
ROBOT_POSE_LOG_DIR = "./data/log"
ROBOT_POSE_LOG_PATH = "./data/log/calculated_robot_poses.log"

# Home joint configuration (radians)
HOME_JOINTS = [0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0]

# Camera pose relative to the robot base (in millimeters)
CAMERA_POSE_X_MM = 500
CAMERA_POSE_Y_MM = 300
CAMERA_POSE_Z_MM = 800

# Camera orientation relative to the robot base (in degrees)
CAMERA_ROLL_DEG = 15.0
CAMERA_PITCH_DEG = -10.0
CAMERA_YAW_DEG = 45.0

# Motion controller safety parameters
APPROACH_HEIGHT_OFFSET = 0.100   # meters - approach height above pick/place position
DEFAULT_VELOCITY = 0.5           # m/s
DEFAULT_ACCELERATION = 0.5       # m/s²
MAX_ABS_CARTESIAN_M = 2.0        # hard safety bound for x/y/z (meters)
FALLBACK_DOWNWARD_ORIENTATION = [0.0, float(np.pi), 0.0]  # axis-angle, radians

# Robot workspace bounds (meters) - UR5e approximate reachable workspace
# These define the Cartesian space we allow moveL commands to target
ROBOT_WORKSPACE_X_MIN = -0.75      # meters (avoid reaching behind base)
ROBOT_WORKSPACE_X_MAX = 0.75      # meters (UR5e reach ~0.85m, with margin)
ROBOT_WORKSPACE_Y_MIN = -0.75     # meters
ROBOT_WORKSPACE_Y_MAX = 0.75      # meters
ROBOT_WORKSPACE_Z_MIN = 0.0       # meters (stay above table/floor)
ROBOT_WORKSPACE_Z_MAX = 0.75      # meters (comfortable work height)

