"""Defines the data models for camera detections of boxes."""

from pydantic import BaseModel

class BoxPose(BaseModel):
    """Represents a single box pose."""

    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float = 0.0


class CameraDetections(BaseModel):
    """Represents the camera detections of boxes."""
    
    description: str
    detections: list[BoxPose]