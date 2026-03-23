"""
Unit tests for coordinate transformation math.

Verifies camera_to_robot / robot_to_camera / build_rotation_matrix /
build_homogeneous_transform with known input/output pairs.

Run with: pytest tests/test_transformation.py -v
"""

import numpy as np
import pytest

from transforms.coordinate import (
    build_rotation_matrix,
    build_homogeneous_transform,
    camera_to_robot,
    robot_to_camera,
)
from config.config import (
    CAMERA_POSE_X_MM,
    CAMERA_POSE_Y_MM,
    CAMERA_POSE_Z_MM,
    CAMERA_ROLL_DEG,
    CAMERA_PITCH_DEG,
    CAMERA_YAW_DEG,
)


def _expected_transform() -> np.ndarray:
    """Build the reference 4x4 camera-to-robot transform from config."""
    t = np.array([CAMERA_POSE_X_MM, CAMERA_POSE_Y_MM, CAMERA_POSE_Z_MM], dtype=float)
    rotation_matrix = build_rotation_matrix(
        np.deg2rad(CAMERA_ROLL_DEG),
        np.deg2rad(CAMERA_PITCH_DEG),
        np.deg2rad(CAMERA_YAW_DEG),
    )
    return build_homogeneous_transform(rotation_matrix, t)


class TestBuildRotationMatrix:
    def test_zero_angles_gives_identity(self):
        rotation_matrix = build_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(rotation_matrix, np.eye(3))

    def test_output_shape(self):
        rotation_matrix = build_rotation_matrix(0.1, -0.2, 0.3)
        assert rotation_matrix.shape == (3, 3)

    def test_is_orthogonal(self):
        """rotation_matrix @ rotation_matrix.T must equal the identity (orthogonality)."""
        rotation_matrix = build_rotation_matrix(
            np.deg2rad(CAMERA_ROLL_DEG),
            np.deg2rad(CAMERA_PITCH_DEG),
            np.deg2rad(CAMERA_YAW_DEG),
        )
        np.testing.assert_array_almost_equal(rotation_matrix @ rotation_matrix.T, np.eye(3))

    def test_determinant_is_one(self):
        """A proper rotation matrix must have det = +1."""
        rotation_matrix = build_rotation_matrix(
            np.deg2rad(CAMERA_ROLL_DEG),
            np.deg2rad(CAMERA_PITCH_DEG),
            np.deg2rad(CAMERA_YAW_DEG),
        )
        assert abs(np.linalg.det(rotation_matrix) - 1.0) < 1e-9

    def test_pure_yaw_90_degrees(self):
        """90° yaw: X → Y, Y → -X, Z unchanged."""
        rotation_matrix = build_rotation_matrix(0.0, 0.0, np.pi / 2)
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        np.testing.assert_array_almost_equal(rotation_matrix, expected)

    def test_pure_pitch_90_degrees(self):
        """90° pitch: Z → X, X → -Z, Y unchanged."""
        rotation_matrix = build_rotation_matrix(0.0, np.pi / 2, 0.0)
        expected = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        np.testing.assert_array_almost_equal(rotation_matrix, expected)

    def test_pure_roll_90_degrees(self):
        """90° roll: Y → Z, Z → -Y, X unchanged."""
        rotation_matrix = build_rotation_matrix(np.pi / 2, 0.0, 0.0)
        expected = np.array([
            [1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0],
        ])
        np.testing.assert_array_almost_equal(rotation_matrix, expected)


class TestBuildHomogeneousTransform:
    def test_output_shape(self):
        rotation_matrix = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        transformation_matrix = build_homogeneous_transform(rotation_matrix, t)
        assert transformation_matrix.shape == (4, 4)

    def test_identity_rotation_translation_embedded(self):
        t = np.array([10.0, 20.0, 30.0])
        transformation_matrix = build_homogeneous_transform(np.eye(3), t)
        np.testing.assert_array_equal(transformation_matrix[:3, :3], np.eye(3))
        np.testing.assert_array_equal(transformation_matrix[:3, 3], t)
        assert transformation_matrix[3, 3] == 1.0
        np.testing.assert_array_equal(transformation_matrix[3, :3], [0.0, 0.0, 0.0])

    def test_rotation_block_is_set(self):
        rotation_matrix = build_rotation_matrix(0.1, 0.2, 0.3)
        t = np.array([5.0, -5.0, 0.0])
        transformation_matrix = build_homogeneous_transform(rotation_matrix, t)
        np.testing.assert_array_equal(transformation_matrix[:3, :3], rotation_matrix)
        np.testing.assert_array_equal(transformation_matrix[:3, 3], t)


class TestCameraToRobot:
    def test_origin_maps_to_camera_pose(self):
        """
        A point at the camera origin [0,0,0] must map exactly to the
        camera's position in the robot frame (pure translation).
        """
        result = camera_to_robot(np.array([0.0, 0.0, 0.0]))
        expected = np.array([CAMERA_POSE_X_MM, CAMERA_POSE_Y_MM, CAMERA_POSE_Z_MM])
        np.testing.assert_array_almost_equal(result, expected)

    def test_known_point_x_axis(self):
        """
        Known pair: offset along camera X axis only.
        Expected = transformation_matrix @ [100, 0, 0, 1] using the reference transform.
        """
        transformation_matrix = _expected_transform()
        point = np.array([100.0, 0.0, 0.0])
        expected = (transformation_matrix @ np.append(point, 1.0))[:3]
        result = camera_to_robot(point)
        np.testing.assert_array_almost_equal(result, expected)

    def test_known_point_y_axis(self):
        transformation_matrix = _expected_transform()
        point = np.array([0.0, 100.0, 0.0])
        expected = (transformation_matrix @ np.append(point, 1.0))[:3]
        result = camera_to_robot(point)
        np.testing.assert_array_almost_equal(result, expected)

    def test_known_point_z_axis(self):
        transformation_matrix = _expected_transform()
        point = np.array([0.0, 0.0, 100.0])
        expected = (transformation_matrix @ np.append(point, 1.0))[:3]
        result = camera_to_robot(point)
        np.testing.assert_array_almost_equal(result, expected)

    def test_known_point_general(self):
        """Realistic detection: [50, -30, 0] mm in camera frame."""
        transformation_matrix = _expected_transform()
        point = np.array([50.0, -30.0, 0.0])
        expected = (transformation_matrix @ np.append(point, 1.0))[:3]
        result = camera_to_robot(point)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape(self):
        result = camera_to_robot(np.array([10.0, 20.0, 30.0]))
        assert result.shape == (3,)

    def test_negative_coordinates(self):
        transformation_matrix = _expected_transform()
        point = np.array([-25.0, 80.0, -50.0])
        expected = (transformation_matrix @ np.append(point, 1.0))[:3]
        result = camera_to_robot(point)
        np.testing.assert_array_almost_equal(result, expected)


class TestRobotToCamera:
    def test_camera_pose_maps_to_origin(self):
        """
        The camera's own position in robot frame must map back to [0,0,0].
        """
        robot_point = np.array([CAMERA_POSE_X_MM, CAMERA_POSE_Y_MM, CAMERA_POSE_Z_MM])
        result = robot_to_camera(robot_point)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_known_point(self):
        """
        Expected = T_inv @ [p, 1] using the reference inverse transform.
        """
        transformation_matrix = _expected_transform()
        T_inv = np.linalg.inv(transformation_matrix)
        robot_point = np.array([540.0, 285.0, 810.0])
        expected = (T_inv @ np.append(robot_point, 1.0))[:3]
        result = robot_to_camera(robot_point)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape(self):
        result = robot_to_camera(np.array([500.0, 300.0, 800.0]))
        assert result.shape == (3,)


class TestRoundTrip:
    @pytest.mark.parametrize("point", [
        [0.0, 0.0, 0.0],
        [50.0, -30.0, 0.0],
        [120.0, 45.0, 0.0],
        [-25.0, 80.0, 0.0],
        [90.0, -60.0, 500.0],
        [-200.0, 150.0, -100.0],
    ])
    def test_camera_robot_camera_roundtrip(self, point):
        """camera_to_robot → robot_to_camera must recover original point."""
        p = np.array(point, dtype=float)
        result = robot_to_camera(camera_to_robot(p))
        np.testing.assert_array_almost_equal(result, p, decimal=6)

    @pytest.mark.parametrize("point", [
        [500.0, 300.0, 800.0],
        [600.0, 250.0, 750.0],
        [450.0, 350.0, 900.0],
    ])
    def test_robot_camera_robot_roundtrip(self, point):
        """robot_to_camera → camera_to_robot must recover original point."""
        p = np.array(point, dtype=float)
        result = camera_to_robot(robot_to_camera(p))
        np.testing.assert_array_almost_equal(result, p, decimal=6)
