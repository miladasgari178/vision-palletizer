"""Palletizer State Machine which manages the lifecycle of palletizing operations."""

import json
import numpy as np
from datetime import datetime
import os
import threading

from state_machine.core import StateMachine, BaseTriggers
from state_machine.decorators import on_enter_state, on_state_change
from config.config import DETECTED_BOX_POSITION_PATH, ROBOT_POSE_LOG_PATH
from palletizer.grid import calculate_place_positions
from palletizer.state_machine_model import (
    PalletizerState,
    States,
    PalletizerContext,
    TRANSITIONS,
)
from robot.motion import MotionController
from transforms.coordinate import camera_to_robot
from data.camera_detection import CameraDetections



class PalletizerStateMachine(StateMachine):
    """State machine for palletizing operations.

    Args:
        motion_controller: An instance of MotionController to execute robot movements.
    """
    
    def __init__(self, motion_controller: MotionController):
        super().__init__(
            states=States,
            transitions=TRANSITIONS,
            enable_last_state_recovery=False,
        )
        self.motion_controller: MotionController = motion_controller
        self.context = PalletizerContext()
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Request graceful stop at the next safe transition point."""
        self._stop_requested.set()

    def clear_stop_request(self) -> None:
        """Clear pending stop request before starting a new cycle."""
        self._stop_requested.clear()

    def _stop_if_requested(self) -> bool:
        """Stop machine if a stop was requested."""
        if not self._stop_requested.is_set():
            return False

        # If already idle, consume the request and consider it handled.
        if self.current_state == PalletizerState.IDLE:
            self._stop_requested.clear()
            return True

        try:
            self.trigger("stop")
            return True
        except Exception:
            return False
        finally:
            if self.current_state == PalletizerState.IDLE:
                self._stop_requested.clear()

    def _trigger_or_ignore_if_idle(self, event: str) -> bool:
        """
        Trigger a state-machine event, unless machine is already IDLE.
            This avoids faulting when a concurrent stop moved the machine to
            `ready` while an on_enter handler was about to emit a completion event.

        """
        try:
            self.trigger(event)
            return True
        except Exception:
            if self.current_state == PalletizerState.IDLE:
                self.clear_stop_request()
                print(f"ℹ Ignoring '{event}' because palletizer is already IDLE")
                return False
            raise

    def _append_pose_log(self, stage: str, pose_mm: list[float]) -> None:
        """Append a calculated robot pose entry to the pose log file."""
        os.makedirs(os.path.dirname(os.path.abspath(ROBOT_POSE_LOG_PATH)), exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "stage": stage,
            "box_index": self.context.current_box_index,
            "state": self.current_state.name,
            "pose_mm": {
                "x": float(pose_mm[0]),
                "y": float(pose_mm[1]),
                "z": float(pose_mm[2]),
            }
        }

        # Append the new entry as a log entry in the log file
        with open(ROBOT_POSE_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")
    
    @property
    def current_state(self) -> PalletizerState:
        """Get current state."""
        state_str = self.state
        mapping = {
            "ready": PalletizerState.IDLE,
            "fault": PalletizerState.FAULT,
            "Running_homing": PalletizerState.HOMING,
            "Running_picking": PalletizerState.PICKING,
            "Running_placing": PalletizerState.PLACING,
        }
        return mapping.get(state_str, PalletizerState.IDLE)
    
    @property
    def progress(self) -> dict:
        """Get current progress state for API responses."""
        return {
            "state": self.current_state.name,
            "current_box": self.context.current_box_index,
            "total_boxes": self.context.total_boxes,
            "error": self.context.error_message if self.context.error_message else None,
        }
    
    def is_configured(self) -> bool:
        """Check if palletizer is configured with a valid grid."""
        return self.context.total_boxes > 0

    def configure(
        self,
        rows: int,
        cols: int,
        box_size_mm: tuple[float, float, float],
        pallet_origin_mm: tuple[float, float, float],
    ) -> bool:
        """Configure palletizing parameters. Only valid in IDLE state."""
        if self.current_state != PalletizerState.IDLE:
            return False
        
        self.context.rows = rows
        self.context.cols = cols
        self.context.box_size_mm = box_size_mm
        self.context.pallet_origin_mm = pallet_origin_mm
        self.context.total_boxes = rows * cols
        self.context.current_box_index = 0
        self.context.place_positions = []
        return True
    
    def begin(self) -> bool:
        """Start the palletizing sequence."""
        if self.current_state != PalletizerState.IDLE:
            return False
        try:
            self.clear_stop_request()
            self.trigger("start")
            return True
        except Exception:
            return False
    
    def stop(self) -> bool:
        """Stop the palletizing sequence and return to IDLE."""
        if self.current_state == PalletizerState.IDLE:
            print("✓ Palletizer already in IDLE state")
            self.clear_stop_request()
            return True
        try:
            print("⚠ Stopping palletizer sequence...")
            self.request_stop()
            try:
                self.trigger("stop")
                # Immediate stop succeeded; consume request.
                self.clear_stop_request()
            except Exception:
                # If currently inside an active transition, on_enter handlers
                # will observe the stop request and stop at the next boundary.
                pass
            return True
        except Exception:
            return False
    
    def reset(self) -> bool:
        """Reset from FAULT state to IDLE."""
        try:
            self.trigger(BaseTriggers.RESET.value)
            self.context.error_message = ""
            return True
        except Exception:
            return False
    
    
    def fault(self, message: str) -> bool:
        """Transition to FAULT state with an error message."""
        self.context.error_message = message
        try:
            self.trigger(BaseTriggers.TO_FAULT.value)
            return True
        except Exception:
            return False
            
        
    def build_grid_positions(self):
        """Pre-compute place positions based on grid configuration."""
        self.place_positions = calculate_place_positions(
            rows=self.context.rows,
            cols=self.context.cols,
            box_size_mm=self.context.box_size_mm,
            pallet_origin_mm=self.context.pallet_origin_mm,
        )

    @on_enter_state(States.running.homing)
    def on_enter_homing(self, _):
        """Execute homing sequence: move robot to home position."""
        try:
            if self._stop_if_requested():
                return
            self.motion_controller.move_to_home()
            print("✓ Homing completed successfully")
            if self._stop_if_requested():
                return
            self._trigger_or_ignore_if_idle("finished_homing")
        except Exception as error:
            self.fault(f"Homing failed: {str(error)}")

    
    @on_enter_state(States.running.picking)
    def on_enter_picking(self, _):
        """Execute pick sequence based on camera detections"""

        try:
            if self._stop_if_requested():
                return

            if self.context.total_boxes <= 0:
                self.fault("Pick failed: palletizer is not configured")
                return

            if self.context.current_box_index >= self.context.total_boxes:
                self.fault("Pick failed: current_box_index out of range")
                return

            # Load camera detections from file
            with open (DETECTED_BOX_POSITION_PATH, "r") as file:
                box_places = json.load(file)
                camera_detections = CameraDetections(**box_places)
            
            detections = camera_detections.detections
            if not detections:
                self.fault("Pick failed: no camera detections available")
                return

            # Reuse detections if configured grid has more boxes than mocked detections
            idx = self.context.current_box_index % len(detections)
            det = detections[idx]

            # Keep transform inputs in mm (camera_to_robot expects mm)
            cam_x_mm = float(det.x_mm)
            cam_y_mm = float(det.y_mm)
            cam_z_mm = float(det.z_mm)

            # Transform camera coordinates to robot frame (mm)
            robot_pick_mm = camera_to_robot(np.array([cam_x_mm, cam_y_mm, cam_z_mm]))

            # Motion controller expects meters
            self._append_pose_log(
                stage="pick_calculated",
                pose_mm=[float(robot_pick_mm[0]), float(robot_pick_mm[1]), float(robot_pick_mm[2])]
            )
            
            # Convert to meters for motion controller
            robot_pick = robot_pick_mm / 1000.0
            self.context.pick_position = tuple(robot_pick)

            # Execute pick motion
            motion_completed = self.motion_controller.move_to_pick(position=list(float(x) for x in robot_pick))
            if not motion_completed:
                self.fault("Pick failed: motion controller reported failure")
                return

            if self._stop_if_requested():
                return
            self._trigger_or_ignore_if_idle("finished_picking")

        except Exception as error:
            self.fault(f"Pick failed: {str(error)}")
    
    @on_enter_state(States.running.placing)
    def on_enter_placing(self, _):
        """Execute place sequence to the next position in the grid"""
        try:
            if self._stop_if_requested():
                return

            if self.context.total_boxes <= 0:
                self.fault("Place failed: palletizer is not configured")
                return

            if self.context.current_box_index >= self.context.total_boxes:
                self.fault("Place failed: current_box_index out of range")
                return

            if self.context.pick_position is None:
                self.fault("Place failed: no pick position available")
                return

            # Build grid positions once, in row-major order.
            if not self.context.place_positions:
                # Get the next place position from the pre-computed grid
                self.context.place_positions = calculate_place_positions(
                    rows=self.context.rows,
                    cols=self.context.cols,
                    box_size_mm=self.context.box_size_mm,
                    pallet_origin_mm=self.context.pallet_origin_mm,
                )

            # Get the next place position from the pre-computed grid
            place_position = self.context.place_positions[self.context.current_box_index]

            # Move to place position
            place_position_m = [
                place_position[0] / 1000.0,
                place_position[1] / 1000.0,
                place_position[2] / 1000.0,
            ]
            self._append_pose_log(
                stage="place_calculated",
                pose_mm=[float(place_position[0]), float(place_position[1]), float(place_position[2])],
            )
            motion_completed = self.motion_controller.move_to_place(
                position=place_position_m
            )
            if not motion_completed:
                self.fault("Place failed: motion controller reported failure")
                return

            if self._stop_if_requested():
                return
            self.context.current_box_index += 1
            if self.context.current_box_index >= self.context.total_boxes:
                self._trigger_or_ignore_if_idle("cycle_complete")
            else:
                self._trigger_or_ignore_if_idle("finished_placing")
        except Exception as error:
            self.fault(f"Place failed: {str(error)}")
    
    @on_state_change
    def on_any_state_change(self, old_state: str, new_state: str, trigger: str):
        """Called on every state transition. Useful for logging."""
        print(f"Transition: {old_state} --({trigger})--> {new_state}")
