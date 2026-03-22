"""Models for Palletizer API"""

from pydantic import BaseModel, Field
from typing import Optional


class PalletConfig(BaseModel):
    """Configuration for palletizing operation."""
    
    rows: int = Field(..., ge=1, le=10, description="Number of rows in the grid")
    cols: int = Field(..., ge=1, le=10, description="Number of columns in the grid")
    box_width_mm: float = Field(..., gt=0, description="Box width in mm (X direction)")
    box_depth_mm: float = Field(..., gt=0, description="Box depth in mm (Y direction)")
    box_height_mm: float = Field(..., gt=0, description="Box height in mm (Z direction)")
    pallet_origin_x_mm: float = Field(..., description="Pallet origin X in mm")
    pallet_origin_y_mm: float = Field(..., description="Pallet origin Y in mm")
    pallet_origin_z_mm: float = Field(..., description="Pallet origin Z in mm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rows": 2,
                "cols": 2,
                "box_width_mm": 100.0,
                "box_depth_mm": 100.0,
                "box_height_mm": 50.0,
                "pallet_origin_x_mm": 400.0,
                "pallet_origin_y_mm": -200.0,
                "pallet_origin_z_mm": 100.0,
            }
        }


class VisionDetection(BaseModel):
    """Simulated vision detection of a box."""
    
    x_mm: float = Field(..., description="Box X position in camera frame (mm)")
    y_mm: float = Field(..., description="Box Y position in camera frame (mm)")
    z_mm: float = Field(..., description="Box Z position in camera frame (mm)")
    yaw_deg: Optional[float] = Field(0.0, description="Box rotation about Z (degrees)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x_mm": 50.0,
                "y_mm": -30.0,
                "z_mm": 0.0,
                "yaw_deg": 15.0,
            }
        }


class StatusResponse(BaseModel):
    """Palletizer status response."""
    
    state: str = Field(..., description="Current state machine state")
    current_box: int = Field(..., description="Current box index (0-based)")
    total_boxes: int = Field(..., description="Total boxes to palletize")
    error: Optional[str] = Field(None, description="Error message if in FAULT state")


class ConfigResponse(BaseModel):
    """Configuration response."""
    
    success: bool
    message: str
    grid_size: Optional[str] = None


class CommandResponse(BaseModel):
    """Generic command response."""
    
    success: bool
    message: str


class PositionData(BaseModel):
    """Position data with coordinates."""
    
    x_mm: float = Field(..., description="X coordinate in mm")
    y_mm: float = Field(..., description="Y coordinate in mm")
    z_mm: float = Field(..., description="Z coordinate in mm")
    box_index: int = Field(..., description="Box index (0-based)")


class CalculatedPositionsResponse(BaseModel):
    """Response containing all calculated place positions."""
    
    success: bool = Field(..., description="Indicates if the calculation was successful")
    grid_rows: int = Field(..., description="Number of rows in grid")
    grid_cols: int = Field(..., description="Number of columns in grid")
    positions: list[PositionData] = Field(..., description="List of place positions")
    total_positions: int = Field(..., description="Total number of positions")


class TransformResponse(BaseModel):
    """Transformation result with camera and robot frame coordinates."""
    
    success: bool = Field(..., description="Indicates if the transformation was successful")
    camera_frame: dict = Field(..., description="Coordinates in camera frame (mm)")
    robot_frame: dict = Field(..., description="Coordinates in robot frame (mm)")