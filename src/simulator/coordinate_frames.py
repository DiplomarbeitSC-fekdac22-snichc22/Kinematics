"""Coordinate conversions between the project frame and Webots."""

from __future__ import annotations


def robot_to_webots(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    *,
    top_reference_height_mm: float = 500.0,
) -> tuple[float, float, float]:
    """Convert X-forward/Z-right/Y-down millimetres to Webots metres.

    Webots uses X forward, Y left and Z up. The project's Y value is a
    distance measured downward from the top of the 500 mm workcell.
    """
    return (
        float(x_mm) / 1000.0,
        -float(z_mm) / 1000.0,
        (float(top_reference_height_mm) - float(y_mm)) / 1000.0,
    )


def webots_to_robot(
    x_m: float,
    y_m: float,
    z_m: float,
    *,
    top_reference_height_mm: float = 500.0,
) -> tuple[float, float, float]:
    """Convert Webots X-forward/Y-left/Z-up metres to robot millimetres."""
    return (
        float(x_m) * 1000.0,
        float(top_reference_height_mm) - float(z_m) * 1000.0,
        -float(y_m) * 1000.0,
    )
