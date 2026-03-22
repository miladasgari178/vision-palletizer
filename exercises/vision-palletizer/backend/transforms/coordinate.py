"""Coordinate Transformations which transform coordinates between camera frame and robot base frame."""

import numpy as np
from config.config import (
    CAMERA_POSE_X_MM,
    CAMERA_POSE_Y_MM,
    CAMERA_POSE_Z_MM,
    CAMERA_ROLL_DEG,
    CAMERA_PITCH_DEG,
    CAMERA_YAW_DEG,
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


def camera_to_robot(point_camera: np.ndarray) -> np.ndarray:
    """
    Transform a point from camera frame to robot base frame.
    
    Args:
        point_camera: [x, y, z] coordinates in camera frame (mm)
    
    Returns:
        [x, y, z] coordinates in robot base frame (mm)
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

    return point_robot_h[:3]


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


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to roll, pitch, yaw in radians.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
    
    Returns:
        [roll, pitch, yaw] in radians
    """
    if abs(rotation_matrix[2,0]) < 1.0 - 1e-6:
        pitch = -np.arcsin(rotation_matrix[2,0])
        roll = np.arctan2(rotation_matrix[2,1]/np.cos(pitch), rotation_matrix[2,2]/np.cos(pitch))
        yaw = np.arctan2(rotation_matrix[1,0]/np.cos(pitch), rotation_matrix[0,0]/np.cos(pitch))
    else:
        pitch = np.pi/2 if rotation_matrix[2,0] <= -1.0 else -np.pi/2
        roll = 0.0
        yaw = np.arctan2(-rotation_matrix[0,1], rotation_matrix[1,1])


    return np.array([roll, pitch, yaw], dtype=float)


def camera_to_robot_orientation(camera_orientation: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles from camera frame to robot base frame.
    
    Args:
        camera_orientation: [roll, pitch, yaw] in CAMERA frame (radians)
    
    Returns:
        [roll, pitch, yaw] in ROBOT frame (radians)
    """
    # Rotation of camera in robot frame
    roll = np.deg2rad(CAMERA_ROLL_DEG)
    pitch = np.deg2rad(CAMERA_PITCH_DEG)
    yaw = np.deg2rad(CAMERA_YAW_DEG)
    rotation_matrix_base_camera = build_rotation_matrix(roll, pitch, yaw)

    # Rotation in camera frame
    rotation_matrix_camera_obj = build_rotation_matrix(
        camera_orientation[0],
        camera_orientation[1],
        camera_orientation[2],
    )

    # Transform to robot frame
    rotation_matrix_robot_obj = rotation_matrix_base_camera @ rotation_matrix_camera_obj

    # Extract Euler angles in robot frame
    robot_orientation = rotation_matrix_to_euler(rotation_matrix_robot_obj)
    return robot_orientation
