from pathlib import Path
from typing import Any

from config.config_loader import DEFAULT_CONFIG_DIR, load_config


def shelf_compartment_y_ranges(
        kinematics: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    """Return the clear Y intervals inside the shelf extension."""
    shelf = kinematics["shelving_mm"]
    first_top = float(shelf["first_shelf_from_top"])
    clear_height = float(shelf["compartment_height"])
    pitch = clear_height + float(shelf["floor_thickness"])
    count = int(shelf["compartment_count"])

    return tuple(
        (
            first_top + index * pitch,
            first_top + index * pitch + clear_height,
        )
        for index in range(count)
    )


def workspace_violation_reasons(
        x_mm: float,
        y_mm: float,
        z_mm: float,
        kinematics: dict[str, Any],
) -> list[str]:
    """Describe free-box or shelf-compartment workspace violations."""
    if not kinematics["validation"]["check_workspace_bounds"]:
        return []

    bounds = kinematics["workspace_bounds_robot_base_mm"]
    reasons: list[str] = []

    for axis, value in (("y", y_mm), ("z", z_mm)):
        if not (
                float(bounds[f"{axis}_min"])
                <= value
                <= float(bounds[f"{axis}_max"])
        ):
            reasons.append(
                f"{axis}={value:.1f} mm outside workspace bounds"
            )

    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    if x_min <= x_mm <= x_max:
        return reasons

    shelf = kinematics["shelving_mm"]
    shelf_depth = float(shelf["depth"])
    inside_positive_shelf_depth = (
            shelf["x_direction"] == "positive"
            and x_max < x_mm <= x_max + shelf_depth
    )

    if inside_positive_shelf_depth:
        inside_open_compartment = any(
            top <= y_mm < bottom
            for top, bottom in shelf_compartment_y_ranges(kinematics)
        )
        if not inside_open_compartment:
            reasons.append(
                f"x={x_mm:.1f} mm enters shelf depth but "
                f"y={y_mm:.1f} mm intersects a shelf floor or "
                "closed shelf region"
            )
        return reasons

    reasons.append(f"x={x_mm:.1f} mm outside workspace bounds")
    return reasons


def is_target_inside_workspace_bounds(
        x_mm: float,
        y_mm: float,
        z_mm: float,
        config_dir: Path | str = DEFAULT_CONFIG_DIR,
) -> bool:
    kinematics = load_config("kinematics_settings.toml", config_dir)
    return not workspace_violation_reasons(
        x_mm,
        y_mm,
        z_mm,
        kinematics,
    )


def are_joint_angles_inside_limits(joint_angles_deg: dict[str, float]) -> bool:
    kinematics = load_config("kinematics_settings.toml")
    servo_calibration = load_config("servo_calibration.toml")

    if not kinematics["validation"]["check_joint_limits"]:
        return True

    joints = servo_calibration["joints"]

    for joint_name, angle_deg in joint_angles_deg.items():
        if joint_name not in joints:
            return False

        joint = joints[joint_name]

        if not joint["theta_min_deg"] <= angle_deg <= joint["theta_max_deg"]:
            return False

    return True


def is_target_reachable(
        x_mm: float,
        y_mm: float,
        z_mm: float,
        joint_angles_deg: dict[str, float],
) -> bool:
    if not is_target_inside_workspace_bounds(x_mm, y_mm, z_mm):
        return False

    if not are_joint_angles_inside_limits(joint_angles_deg):
        return False

    return True
