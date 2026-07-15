from math import radians, cos, sin
from pathlib import Path
from typing import Any

from config.config_loader import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def _angle_for_role(
    joint_angles_deg: dict[str, float],
    servo_config: dict[str, Any],
    role: str,
) -> float:
    for joint_name, joint_config in servo_config["joints"].items():
        if joint_config.get("kinematic_role") == role:
            if joint_name not in joint_angles_deg:
                raise KeyError(f"Missing angle for joint {joint_name!r}")

            return float(joint_angles_deg[joint_name])

    raise KeyError(f"No servo joint configured for role {role!r}")


def calculate_gripper_center(
    joint_angles_deg: dict[str, float],
    config_dir: Path | str = CONFIG_DIR,
) -> dict[str, float]:
    """Calculate the commanded gripper-center position."""
    config_dir = Path(config_dir)

    geometry = load_config("robot_geometry.toml", config_dir)
    servo = load_config("servo_calibration.toml", config_dir)
    settings = load_config("kinematics_settings.toml", config_dir)

    links = geometry["link_lengths_mm"]
    model = settings["model"]
    input_coordinates = settings.get("input_coordinates", {})

    l1 = float(links["L1_shoulder_to_elbow"])
    l2 = float(links["L2_elbow_to_wrist"])

    lg = (
        float(links[model["selected_Lg_key"]])
        if model["use_gripper_offset"]
        else 0.0
    )

    h0 = (
        float(links[model["selected_h0_key"]])
        if model["use_h0_from_robot_geometry"]
        else 0.0
    )

    theta1 = radians(
        _angle_for_role(joint_angles_deg, servo, "theta1")
    )
    theta2 = radians(
        _angle_for_role(joint_angles_deg, servo, "theta2")
    )
    theta3 = radians(
        _angle_for_role(joint_angles_deg, servo, "theta3")
    )
    theta4 = radians(
        _angle_for_role(joint_angles_deg, servo, "theta4")
    )

    # J3 is stored as the positive interior angle, not as the signed relative
    # rotation between the two links. The configured sign selects the
    # roof-mounted elbow branch used by IK and Webots.
    elbow_relative_sign = float(
        settings.get("fk", {}).get(
            "elbow_relative_sign",
            -1.0,
        )
    )

    elbow_relative_angle = elbow_relative_sign * (
        radians(180.0) - theta3
    )
    forearm_angle = theta2 + elbow_relative_angle
    gripper_approach_angle = forearm_angle + theta4

    radial_mm = (
        l1 * cos(theta2)
        + l2 * cos(forearm_angle)
        + lg * cos(gripper_approach_angle)
    )

    gripper_y_math_mm = (
        l1 * sin(theta2)
        + l2 * sin(forearm_angle)
        + lg * sin(gripper_approach_angle)
    )

    max_height = float(
        input_coordinates.get("max_height_mm", 0.0)
    )

    base_x, base_y, base_z = input_coordinates.get(
        "base_rotation_axis_at_mounting_plate_mm",
        [0.0, max_height, 0.0],
    )

    base_x = float(base_x)
    base_y = float(base_y)
    base_z = float(base_z)

    y_direction = input_coordinates.get(
        "y_positive_direction",
        "up",
    )

    if y_direction == "down":
        shoulder_y_from_top_mm = base_y + h0
        y_mm = shoulder_y_from_top_mm - gripper_y_math_mm
    elif y_direction == "up":
        y_mm = h0 + gripper_y_math_mm
    else:
        raise ValueError(
            f"Unsupported y_positive_direction: {y_direction!r}"
        )

    return {
        "x_mm": base_x + radial_mm * cos(theta1),
        "y_mm": y_mm,
        "z_mm": base_z + radial_mm * sin(theta1),
    }