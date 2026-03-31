"""Coordinate Transformations which transform coordinates between camera frame and robot base frame."""

import numpy as np
from config.config import (
    CAMERA_POSE_X_MM,
    CAMERA_POSE_Y_MM,
    CAMERA_POSE_Z_MM,
    CAMERA_ROLL_DEG,
    CAMERA_PITCH_DEG,
    CAMERA_YAW_DEG,
    FALLBACK_DOWNWARD_ORIENTATION,
)


def build_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Roll-Pitch-Yaw (Euler) angles.
    
    Args:
        roll: Rotation about X-axis in radians
        pitch: Rotation about Y-axis in radians
        yaw: Rotation about Z-axis in radians
    
    Returns:
        3x3 rotation matrix
    """

    cos_x, sin_x = np.cos(roll), np.sin(roll)
    cos_y, sin_y = np.cos(pitch), np.sin(pitch)
    cos_z, sin_z = np.cos(yaw), np.sin(yaw)

    # Rotation about X (roll)
    rotation_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])

    # Rotation about Y (pitch)
    rotation_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    # Rotation about Z (yaw)
    rotation_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1]
    ])

    # Combined rotation: rotation_matrix = Rz * Ry * Rx (Z-Y-X convention)
    rotation_matrix = rotation_z @ rotation_y @ rotation_x

    return rotation_matrix

def build_homogeneous_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Build a 4x4 homogeneous transformation matrix.
    
    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 or (3,) translation vector
    
    Returns:
        4x4 homogeneous transformation matrix
    """

    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)

    # Insert the rotation and translation into the transformation matrix
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation.flatten()

    return transformation_matrix


def axis_angle_to_rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to a 3x3 rotation matrix."""
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return np.eye(3)

    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an axis-angle rotation vector."""
    trace = float(np.trace(rotation_matrix))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))

    if angle < 1e-12:
        return np.zeros(3, dtype=float)

    if np.isclose(angle, np.pi, atol=1e-6):
        diagonal = np.diag(rotation_matrix)
        axis = np.sqrt(np.maximum((diagonal + 1.0) / 2.0, 0.0))

        if axis[0] > 1e-6:
            axis[1] = rotation_matrix[0, 1] / (2.0 * axis[0])
            axis[2] = rotation_matrix[0, 2] / (2.0 * axis[0])
        elif axis[1] > 1e-6:
            axis[0] = rotation_matrix[0, 1] / (2.0 * axis[1])
            axis[2] = rotation_matrix[1, 2] / (2.0 * axis[1])
        else:
            axis[0] = rotation_matrix[0, 2] / (2.0 * axis[2])
            axis[1] = rotation_matrix[1, 2] / (2.0 * axis[2])

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            axis = axis / axis_norm
        return axis * angle

    axis = np.array([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1],
    ], dtype=float) / (2.0 * np.sin(angle))
    return axis * angle


def camera_to_robot(
    point_camera: np.ndarray,
    cam_yaw_deg: float,
) -> np.ndarray:
    """
    Transform a point and orientation from camera frame to robot base frame.
    
    Args:
        point_camera: [x, y, z] coordinates in camera frame (mm)
        cam_yaw_deg: Yaw angle in camera frame (degrees).
    
    Returns:
        [x, y, z, rx, ry, rz] as 6-element numpy array where:
            x, y, z are coordinates in robot base frame (mm)
            rx, ry, rz are axis-angle orientation values in robot base frame (radians)
    """

    # Camera pose in robot frame
    translation = np.array(
        [CAMERA_POSE_X_MM, CAMERA_POSE_Y_MM, CAMERA_POSE_Z_MM], 
        dtype=float
    )

    # Convert angles from degrees to radians
    roll = np.deg2rad(CAMERA_ROLL_DEG)
    pitch = np.deg2rad(CAMERA_PITCH_DEG)
    yaw = np.deg2rad(CAMERA_YAW_DEG)

    # Build the transformation matrix from camera frame to robot frame
    rotation_matrix = build_rotation_matrix(roll, pitch, yaw)
    transformation_matrix = build_homogeneous_transform(rotation_matrix, translation)

    # Convert point in camera frame to homogeneous coordinates
    point_h = np.append(point_camera, 1.0)

    # Transform the point to robot frame
    point_robot_h = transformation_matrix @ point_h

    point_robot = point_robot_h[:3]

    robot_orientation = camera_to_robot_orientation(cam_yaw_deg)
    
    # Return 6-element array: [x, y, z, rx, ry, rz]
    return np.concatenate([point_robot, robot_orientation])


def robot_to_camera(point_robot: np.ndarray) -> np.ndarray:
    """
    Transform a point from robot base frame to camera frame.
    
    Args:
        point_robot: [x, y, z] coordinates in robot base frame (mm)
    
    Returns:
        [x, y, z] coordinates in camera frame (mm)
    """
    # Camera pose in robot frame
    translation = np.array(
        [CAMERA_POSE_X_MM, CAMERA_POSE_Y_MM, CAMERA_POSE_Z_MM], 
        dtype=float
    )

    # Convert angles from degrees to radians
    roll = np.deg2rad(CAMERA_ROLL_DEG)
    pitch = np.deg2rad(CAMERA_PITCH_DEG)
    yaw = np.deg2rad(CAMERA_YAW_DEG)

    # Build the transformation matrix from camera frame to robot frame
    rotation_matrix = build_rotation_matrix(roll, pitch, yaw)
    transformation_matrix = build_homogeneous_transform(rotation_matrix, translation)

    # Invert the transformation matrix to get robot-to-camera transform
    transformation_matrix_inv = np.linalg.inv(transformation_matrix)

    # Convert point in robot frame to homogeneous coordinates
    point_h = np.append(point_robot, 1.0)

    # Transform the point to camera frame
    point_camera_h = transformation_matrix_inv @ point_h

    return point_camera_h[:3]


def camera_to_robot_orientation(cam_yaw_deg: float) -> np.ndarray:
    """
    Convert camera-frame object yaw into robot TCP axis-angle orientation,
    correctly accounting for camera extrinsics.

    Args:
        cam_yaw_deg: Object yaw in camera frame (degrees)

    Returns:
        [rx, ry, rz] axis-angle orientation in robot frame (radians)
    """

    # 1. Camera → Robot rotation (extrinsics)
    roll = np.deg2rad(CAMERA_ROLL_DEG)
    pitch = np.deg2rad(CAMERA_PITCH_DEG)
    yaw = np.deg2rad(CAMERA_YAW_DEG)

    R_cam_to_robot = build_rotation_matrix(roll, pitch, yaw)

    # 2. Object rotation in camera frame (yaw around camera Z)
    R_obj_cam = build_rotation_matrix(0.0, 0.0, np.deg2rad(cam_yaw_deg))

    # 3. Transform object orientation into robot frame
    R_obj_robot = R_cam_to_robot @ R_obj_cam

    # 4. Downward tool orientation (gripper pointing down)
    R_down = axis_angle_to_rotation_matrix(
        np.array(FALLBACK_DOWNWARD_ORIENTATION, dtype=float)
    )

    # 5. Combine: align tool with object while keeping it downward
    tcp_rotation = R_obj_robot @ R_down

    # 6. Convert to axis-angle for robot
    return rotation_matrix_to_axis_angle(tcp_rotation)
