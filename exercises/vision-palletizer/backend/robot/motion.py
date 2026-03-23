"""Robot motion commands for pick and place operations."""

from typing import Optional

from .connection import RobotConnection
from config.config import (
    HOME_JOINTS,
    APPROACH_HEIGHT_OFFSET,
    DEFAULT_VELOCITY,
    DEFAULT_ACCELERATION,
    MAX_ABS_CARTESIAN_M,
    FALLBACK_DOWNWARD_ORIENTATION,
    ROBOT_WORKSPACE_X_MIN,
    ROBOT_WORKSPACE_X_MAX,
    ROBOT_WORKSPACE_Y_MIN,
    ROBOT_WORKSPACE_Y_MAX,
    ROBOT_WORKSPACE_Z_MIN,
    ROBOT_WORKSPACE_Z_MAX,
)

class MotionController:
    """Controls robot motion for palletizing operations."""
    
    # Safety parameters (from config)
    APPROACH_HEIGHT_OFFSET = APPROACH_HEIGHT_OFFSET
    DEFAULT_VELOCITY = DEFAULT_VELOCITY
    DEFAULT_ACCELERATION = DEFAULT_ACCELERATION
    MAX_ABS_CARTESIAN_M = MAX_ABS_CARTESIAN_M
    FALLBACK_DOWNWARD_ORIENTATION = FALLBACK_DOWNWARD_ORIENTATION

    # Workspace bounds (from config)
    WORKSPACE_X_MIN = ROBOT_WORKSPACE_X_MIN
    WORKSPACE_X_MAX = ROBOT_WORKSPACE_X_MAX
    WORKSPACE_Y_MIN = ROBOT_WORKSPACE_Y_MIN
    WORKSPACE_Y_MAX = ROBOT_WORKSPACE_Y_MAX
    WORKSPACE_Z_MIN = ROBOT_WORKSPACE_Z_MIN
    WORKSPACE_Z_MAX = ROBOT_WORKSPACE_Z_MAX
    
    def __init__(self, connection: RobotConnection):
        """
        Initialize motion controller.
        
        Args:
            connection: Active robot connection instance.
        """
        self.connection = connection
        self._gripper_closed = False
        self._cached_orientation: Optional[list[float]] = None
    
    def is_position_in_workspace(self, position: list[float]) -> bool:
        """
        Validate that a Cartesian position is within robot workspace.
        
        Args:
            position: [x, y, z] in meters
        
        Returns:
            True if position is reachable, False otherwise.
        """
        if len(position) < 3:
            return False
        
        x, y, z = position[0], position[1], position[2]
        
        # Check bounds
        if not (self.WORKSPACE_X_MIN <= x <= self.WORKSPACE_X_MAX):
            print(f"[WORKSPACE] X out of bounds: {x} m (bounds: {self.WORKSPACE_X_MIN}-{self.WORKSPACE_X_MAX})")
            return False
        if not (self.WORKSPACE_Y_MIN <= y <= self.WORKSPACE_Y_MAX):
            print(f"[WORKSPACE] Y out of bounds: {y} m (bounds: {self.WORKSPACE_Y_MIN}-{self.WORKSPACE_Y_MAX})")
            return False
        if not (self.WORKSPACE_Z_MIN <= z <= self.WORKSPACE_Z_MAX):
            print(f"[WORKSPACE] Z out of bounds: {z} m (bounds: {self.WORKSPACE_Z_MIN}-{self.WORKSPACE_Z_MAX})")
            return False
        
        return True
    
    def is_pose_valid_for_motion(self, position: list[float]) -> bool:
        """
        Comprehensive validation of target pose before motion.
        Checks workspace bounds and approach clearance.
        
        Args:
            position: [x, y, z] target position in meters
        
        Returns:
            True if pose is valid and safe, False otherwise.
        """
        # Check main position
        if not self.is_position_in_workspace(position):
            return False
        
        # Check approach position (above the target)
        approach_z = position[2] + self.APPROACH_HEIGHT_OFFSET
        if not (self.WORKSPACE_Z_MIN <= approach_z <= self.WORKSPACE_Z_MAX):
            print(f"[WORKSPACE] Approach Z out of bounds: {approach_z} m")
            return False
        
        return True
    
    def move_to_home(self) -> bool:
        """
        Move robot to home/safe position.
        
        Returns:
            True if move completed successfully.
        """
        result = self._move_joint(HOME_JOINTS)
        if result:
            # Cache TCP orientation after homing for entire pick/place cycle
            self._cached_orientation = None  # clear stale cache
            self._cached_orientation = self.get_default_orientation()
        return result
    
    def move_to_pick(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Execute pick motion sequence.
        
        Args:
            position: [x, y, z] pick position in robot base frame (meters)
            orientation: [rx, ry, rz] tool orientation (axis-angle, radians)
                        If None, use default downward orientation.
        
        Returns:
            True if pick completed successfully.
        """
        # Validate target position is reachable
        if not self.is_pose_valid_for_motion(position):
            print("error: Target pick position is outside workspace")
            return False
        
        if orientation is None:
            orientation = self.get_default_orientation()
        
        # Unpack position and orientation for clarity
        x, y, z = position
        rx, ry, rz = orientation

        # Approach from above
        approach_pose = [x, y, z + self.APPROACH_HEIGHT_OFFSET, rx, ry, rz]
        if not self._move_linear(approach_pose):
            print("error: Failed to move to approach pose while picking")
            return False
        
        # Descend to pick
        pick_pose = [x, y, z, rx, ry, rz]
        if not self._move_linear(pick_pose):
            print("error: Failed to move to pick pose")
            return False

        # Close gripper
        if not self.close_gripper():
            print("error: Failed to close gripper")
            return False

        # Retract
        if not self._move_linear(approach_pose):
            print("error: Failed to retract after picking")
            return False

        return True
    
    def move_to_place(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Execute place motion sequence.
        
        Args:
            position: [x, y, z] place position in robot base frame (meters)
            orientation: [rx, ry, rz] tool orientation (axis-angle, radians)
                        If None, use default downward orientation.
        
        Returns:
            True if place completed successfully.
        """
        # Validate target position is reachable
        if not self.is_pose_valid_for_motion(position):
            print("error: Target place position is outside workspace")
            return False
        
        if orientation is None:
            orientation = self.get_default_orientation()
        
        # Unpack position and orientation for clarity
        x, y, z = position
        rx, ry, rz = orientation

        # Approach from above
        approach_pose = [x, y, z + self.APPROACH_HEIGHT_OFFSET, rx, ry, rz]
        if not self._move_linear(approach_pose):
            print("error: Failed to move to approach pose while placing")
            return False

        # Descend to place
        place_pose = [x, y, z, rx, ry, rz]
        if not self._move_linear(place_pose):
            print("error: Failed to move to place pose")
            return False

        # Open gripper
        if not self.open_gripper():
            print("error: Failed to open gripper")
            return False

        # Retract
        if not self._move_linear(approach_pose):
            print("error: Failed to retract after placing")
            return False

        return True
    
    def open_gripper(self) -> bool:
        """
        Open the gripper to release object.
        
        Returns:
            True if gripper opened successfully.
        """
        self._gripper_closed = False
        print("[MOCK] Gripper opened")
        return True
    
    def close_gripper(self) -> bool:
        """
        Close the gripper to grasp object.
        
        Returns:
            True if gripper closed successfully.
        """
        self._gripper_closed = True
        print("[MOCK] Gripper closed")
        return True
    
    def _move_linear(
        self,
        pose: list[float],
        velocity: float = DEFAULT_VELOCITY,
        acceleration: float = DEFAULT_ACCELERATION,
    ) -> bool:
        """
        Execute linear move to target pose.
        
        Args:
            pose: [x, y, z, rx, ry, rz] target pose
            velocity: Move velocity in m/s
            acceleration: Move acceleration in m/s²
        
        Returns:
            True if move completed.
        """
        if any(abs(coord) > self.MAX_ABS_CARTESIAN_M for coord in pose[:3]):
            print(f"[ERROR] Refusing moveL to out-of-range pose {pose[:3]} m")
            return False

        if self.connection.is_mock_mode():
            print(f"[MOCK] moveL to {pose[:3]}")
            return True

        # Real robot
        self.connection.ensure_connected()
        rtde_c = self.connection._rtde_c
        if rtde_c is None:
            raise RuntimeError("RTDE control interface not available")
        
        try:
            # Execute linear move
            rtde_c.moveL(pose, velocity, acceleration)
            return True
        except Exception as e:
            print(f"[ERROR] Failed moveL to {pose[:3]}: {e}")
            return False
    
    def _move_joint(
        self,
        joints: list[float],
        velocity: float = 1.0,
        acceleration: float = 1.0,
    ) -> bool:
        """
        Execute joint move to target configuration.
        
        Args:
            joints: List of 6 joint angles in radians
            velocity: Joint velocity in rad/s
            acceleration: Joint acceleration in rad/s²
        
        Returns:
            True if move completed.
        """
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveJ to {joints}")
            return True
        else:
            self.connection.ensure_connected()
            rtde_c = self.connection._rtde_c
            if rtde_c is None:
                raise RuntimeError("RTDE control interface not available")
            else:
                try:
                    rtde_c.moveJ(joints, velocity, acceleration)
                except Exception as e:
                    print(f"[ERROR] Failed moveJ: {e}")
                    return False
                
                return True
            
    
    def get_default_orientation(self) -> list[float]:
        """
        Get default tool orientation for picking/placing.
        
        Returns:
            [rx, ry, rz] in axis-angle representation.
        
        For real robot operation, reusing the current TCP orientation is
        safer because it avoids sudden orientation flips that can push the
        arm near wrist singularities. If unavailable, use downward fallback.
        """
        # Return cached orientation from home if available (prevents mid-cycle wrist flips)
        if self._cached_orientation is not None:
            return self._cached_orientation
        
        if self.connection.is_mock_mode():
            return self.FALLBACK_DOWNWARD_ORIENTATION

        try:
            tcp_pose = self.connection.get_tcp_pose()
            if len(tcp_pose) >= 6:
                return [float(tcp_pose[3]), float(tcp_pose[4]), float(tcp_pose[5])]
        except Exception:
            pass

        return self.FALLBACK_DOWNWARD_ORIENTATION
