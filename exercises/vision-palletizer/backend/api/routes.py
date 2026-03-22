"""FastAPI routes for palletizer control."""

import json
import os
import threading

from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np

from robot.motion import MotionController
from palletizer.grid import calculate_place_positions
from transforms.coordinate import camera_to_robot
from config.config import DETECTED_BOX_POSITION_PATH
from data.camera_detection import BoxPose, CameraDetections
from api.models import (
    PalletConfig,
    VisionDetection,
    StatusResponse,
    ConfigResponse,
    CommandResponse,
    PositionData,
    CalculatedPositionsResponse,
    TransformResponse,
)
from palletizer.state_machine import PalletizerStateMachine


router = APIRouter()


# Shared in-memory controller instances
motion: Optional[MotionController] = None
palletizer: Optional[PalletizerStateMachine] = None
palletizer_thread: Optional[threading.Thread] = None



@router.post("/configure", response_model=ConfigResponse)
async def configure_palletizer(config: PalletConfig):
    """
    Configure the palletizing operation. It sets up the grid dimensions, 
        box size, and pallet origin. Can only be called when the palletizer
        is in IDLE state.

    Args:
        config: PalletConfig object containing grid dimensions, box size, and pallet origin.
    
    """

    global palletizer, motion

    from main import get_robot_connection

    # Initialize if needed
    if palletizer is None:
        connection = get_robot_connection()
        if connection is None:
            raise HTTPException(status_code=500, detail="Robot connection not available")

        motion = MotionController(connection)
        palletizer = PalletizerStateMachine(motion_controller=motion)

    configured = palletizer.configure(
        rows=config.rows,
        cols=config.cols,
        box_size_mm=(config.box_width_mm, config.box_depth_mm, config.box_height_mm),
        pallet_origin_mm=(
            config.pallet_origin_x_mm,
            config.pallet_origin_y_mm,
            config.pallet_origin_z_mm,
        ),
    )

    if not configured:
        raise HTTPException(
            status_code=409,
            detail="Cannot configure while palletizer is not in IDLE state",
        )

    return ConfigResponse(
        success=True,
        message="Palletizer configured successfully",
        grid_size=f"{config.rows}x{config.cols}",
    )


@router.post("/start", response_model=CommandResponse)
async def start_palletizer():
    """
    Start the palletizing sequence. Begins the pick-and-place cycle. 
        The palletizer must be configured first.

    """
    global palletizer, palletizer_thread

    if palletizer is None:
        raise HTTPException(status_code=400, detail="Palletizer not configured")
    elif not palletizer.is_configured():
        raise HTTPException(status_code=400, detail="Palletizer not configured")
    elif palletizer_thread is not None and palletizer_thread.is_alive():
        raise HTTPException(status_code=409, detail="Palletizer is already running")

    machine = palletizer
    def _run_palletizer_sequence():
        machine.begin()

    # Start the sequence in a background thread so API remains responsive.
    palletizer_thread = threading.Thread(target=_run_palletizer_sequence, daemon=True)
    palletizer_thread.start()
    success = True

    if not success:
        raise HTTPException(status_code=400, detail="Failed to start palletizer")

    return CommandResponse(success=True, message="Palletizer started")


@router.post("/stop", response_model=CommandResponse)
async def stop_palletizer():
    """Stop the palletizing sequence. Gracefully stops the operation and returns to IDLE state"""

    global palletizer, palletizer_thread
    if palletizer is None:
        raise HTTPException(status_code=400, detail="Palletizer not configured")

    # Stop the sequence
    success = palletizer.stop()
    if not success:
        raise HTTPException(status_code=400, detail="Failed to stop palletizer")

    # Let the background worker finish gracefully after stop request.
    if palletizer_thread is not None and palletizer_thread.is_alive():
        palletizer_thread.join(timeout=0.5)

    return CommandResponse(success=True, message="Palletizer stopped")


@router.post("/reset", response_model=CommandResponse)
async def reset_palletizer():
    """Reset from FAULT state. Clears the fault and returns to IDLE state."""

    global palletizer
    if palletizer is None:
        raise HTTPException(status_code=400, detail="Palletizer not configured")

    was_idle_before = palletizer.current_state.name == "IDLE"

    # If not idle, clear fault first.
    if not was_idle_before:
        success = palletizer.reset()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset palletizer")

    # Reset endpoint should always send robot to home.
    homed = palletizer.motion_controller.move_to_home()
    if not homed:
        raise HTTPException(status_code=500, detail="Palletizer reset, but failed to move robot to home")

    if was_idle_before:
        return CommandResponse(success=True, message="Palletizer already IDLE; robot moved to home")

    return CommandResponse(success=True, message="Palletizer reset and robot moved to home")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current palletizer status"""

    global palletizer
    if palletizer is None:
        return StatusResponse(
            state="IDLE",
            current_box=0,
            total_boxes=0,
            error=None,
        )

    progress = palletizer.progress
    return StatusResponse(
        state=progress["state"],
        current_box=progress["current_box"],
        total_boxes=progress["total_boxes"],
        error=progress["error"],
    )


@router.post("/vision/detect", response_model=CommandResponse)
async def simulate_vision_detection(detection: VisionDetection):
    """
    Simulate a vision detection event.
        In a real system, this would come from the vision system.
        For this exercise, use this endpoint to simulate box detections.
        The coordinates are in the camera frame and must be transformed to 
        the robot frame before use.
        
    """
    try:
        new_box = BoxPose(
            x_mm=detection.x_mm,
            y_mm=detection.y_mm,
            z_mm=detection.z_mm,
            yaw_deg=detection.yaw_deg or 0.0,
        )

        # Load existing detections or start fresh
        if os.path.exists(DETECTED_BOX_POSITION_PATH):
            with open(DETECTED_BOX_POSITION_PATH, "r") as f:
                data = json.load(f)
            camera_detections = CameraDetections(**data)
            camera_detections.detections.append(new_box)
        else:
            camera_detections = CameraDetections(
                description="Simulated vision detections",
                detections=[new_box],
            )

        # Persist updated detections so the state machine can read them
        os.makedirs(os.path.dirname(os.path.abspath(DETECTED_BOX_POSITION_PATH)), exist_ok=True)
        with open(DETECTED_BOX_POSITION_PATH, "w") as f:
            json.dump(camera_detections.model_dump(), f, indent=2)

        return CommandResponse(
            success=True,
            message=f"Detection recorded ({len(camera_detections.detections)} total)",
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to record detection: {str(error)}")


@router.get("/debug/positions", response_model=CalculatedPositionsResponse)
async def get_calculated_positions():
    """
    Get all calculated place positions.
        Useful for verifying grid calculations without running the full sequence.
    """

    global palletizer
    if palletizer is None or not palletizer.is_configured():
        raise HTTPException(status_code=400, detail="Palletizer not configured")
    
    positions = calculate_place_positions(
        rows=palletizer.context.rows,
        cols=palletizer.context.cols,
        box_size_mm=palletizer.context.box_size_mm,
        pallet_origin_mm=palletizer.context.pallet_origin_mm,
    )

    # Convert to PositionData objects
    position_data = [
        PositionData(
            x_mm=pos[0],
            y_mm=pos[1],
            z_mm=pos[2],
            box_index=idx,
        )
        for idx, pos in enumerate(positions)
    ]

    return CalculatedPositionsResponse(
        success=True,
        grid_rows=palletizer.context.rows,
        grid_cols=palletizer.context.cols,
        positions=position_data,
        total_positions=len(position_data),
    )


@router.post("/debug/transform", response_model=TransformResponse)
async def test_transform(detection: VisionDetection):
    """
    Test coordinate transformation.
        Transforms the input coordinates and returns both camera and robot frame values.
        Useful for verifying transformation math.
    """
    try:
        # Transform expects camera coordinates in mm
        camera_point_mm = np.array([
            detection.x_mm,
            detection.y_mm,
            detection.z_mm,
        ])
        
        # Transform to robot frame
        robot_point_mm = camera_to_robot(camera_point_mm)
        
        return TransformResponse(
            success=True,
            camera_frame={
                "x_mm": detection.x_mm,
                "y_mm": detection.y_mm,
                "z_mm": detection.z_mm,
            },
            robot_frame={
                "x_mm": float(robot_point_mm[0]),
                "y_mm": float(robot_point_mm[1]),
                "z_mm": float(robot_point_mm[2]),
            },
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transformation failed: {str(error)}",
        )
