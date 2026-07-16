"""Simple inverse kinematics for the configured robot arm.

Math model:
- Runtime X is depth/forward and Z is lateral/right.
- J1 turns the target into the arm plane: theta1 = atan2(z - base_z, x - base_x).
- The arm plane uses r = sqrt(x^2 + z^2) and converts input y from
  "distance downward from the top" to math-y, where up is positive.
- The wrist target is the gripper target minus Lg along the approach angle.
- J2/J3 are solved as a 2-link triangle using the cosine rule.
  Internally the forearm direction is shoulder_math + elbow_relative_math.
- Output J2 uses the same positive-up mathematical convention as the CAD geometry,
  servo limits, and workspace model. J3 is the positive interior elbow bend.
- J4 cancels the shoulder-plus-forearm direction to preserve the configured approach angle.
"""

from __future__ import annotations

import sys
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from pathlib import Path
from typing import Any

from kinematics.angle_to_pwm import (
    angle_to_pwm,
    angle_to_pwm_unclamped,
    clamp_angle,
)
from kinematics.workspace_checker import workspace_violation_reasons

ROOT = Path(__file__).resolve().parents[2]

from config.config_loader import load_config

CONFIG_DIR = ROOT / "configs"


def _joint_for_role(servo_config: dict[str, Any], role: str) -> tuple[str, dict[str, Any]]:
    for joint_name, joint in servo_config["joints"].items():
        if joint.get("kinematic_role") == role:
            return joint_name, joint
    raise KeyError(f"No servo joint configured for role {role!r}")


def calculate_angles(x_mm: float, y_mm: float, z_mm: float, config_dir: Path | str = CONFIG_DIR) -> dict[str, Any]:
    """Return J1-J4 angles, PWM estimates, and reachability for one XYZ target."""
    config_dir = Path(config_dir)
    geometry = load_config("robot_geometry.toml", config_dir)
    servo = load_config("servo_calibration.toml", config_dir)
    settings = load_config("kinematics_settings.toml", config_dir)

    links = geometry["link_lengths_mm"]
    ik = settings["ik"]
    model = settings["model"]
    input_coordinates = settings.get("input_coordinates", {})
    validation = settings["validation"]

    l1 = links["L1_shoulder_to_elbow"]
    l2 = links["L2_elbow_to_wrist"]
    lg = links[model["selected_Lg_key"]] if model["use_gripper_offset"] else 0.0
    h0 = links[model["selected_h0_key"]] if model["use_h0_from_robot_geometry"] else 0.0
    approach = radians(ik["default_approach_angle_deg"])

    # Base rotation and 2D arm-plane coordinates.
    max_height = input_coordinates.get("max_height_mm", 0.0)
    base_origin = input_coordinates.get(
        "base_rotation_axis_at_mounting_plate_mm", [0.0, max_height, 0.0]
    )
    base_x, base_y, base_z = base_origin
    target_x = x_mm - base_x
    target_z = z_mm - base_z
    theta1 = degrees(atan2(target_z, target_x))
    radial = hypot(target_x, target_z)
    wrist_r = radial - lg * cos(approach)
    y_direction = input_coordinates.get("y_positive_direction", "up")
    if y_direction == "down":
        # Input y is a distance from the top, but the triangle math uses up as positive.
        shoulder_y_from_top = base_y + h0
        target_y = shoulder_y_from_top - y_mm
    elif y_direction == "up":
        target_y = y_mm - h0
    else:
        raise ValueError(f"Unsupported y_positive_direction: {y_direction!r}")
    wrist_y = target_y - lg * sin(approach)

    # Cosine rule for the internal relative elbow angle.
    d = hypot(wrist_r, wrist_y)
    cos_theta3 = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if ik["clamp_cosine_rule_argument"]:
        cos_theta3 = clamp_angle(cos_theta3, -1.0, 1.0)

    def solve_branch(elbow_sign: float) -> tuple[float, float, float, float]:
        elbow_rad = atan2(
            elbow_sign * sqrt(max(0.0, 1.0 - cos_theta3 * cos_theta3)),
            cos_theta3,
        )
        shoulder_rad = atan2(wrist_y, wrist_r) - atan2(
            l2 * sin(elbow_rad), l1 + l2 * cos(elbow_rad)
        )
        elbow_r = l1 * cos(shoulder_rad)
        elbow_y = l1 * sin(shoulder_rad)
        return shoulder_rad, elbow_rad, elbow_r, elbow_y

    elbow_sign = float(
        settings.get("fk", {}).get(
            "elbow_relative_sign",
            -1.0,
        )
    )

    theta2_rad, theta3_rad, elbow_r, _ = solve_branch(
        elbow_sign
    )

    theta2 = degrees(theta2_rad)
    theta3 = 180.0 - abs(degrees(theta3_rad))
    theta4 = degrees(approach - theta2_rad - theta3_rad)

    joint_name_1, joint_1 = _joint_for_role(servo, "theta1")
    joint_name_2, joint_2 = _joint_for_role(servo, "theta2")
    joint_name_3, joint_3 = _joint_for_role(servo, "theta3")
    joint_name_4, joint_4 = _joint_for_role(servo, "theta4")
    angles = {
        joint_name_1: theta1,
        joint_name_2: theta2,
        joint_name_3: theta3,
        joint_name_4: theta4,
    }

    reasons: list[str] = []
    min_reach = abs(l1 - l2) + ik["minimum_reach_margin_mm"]
    max_reach = l1 + l2 - ik["maximum_reach_margin_mm"]
    if validation["check_reachability"] and not (min_reach <= d <= max_reach):
        reasons.append(f"wrist distance {d:.1f} mm outside {min_reach:.1f}-{max_reach:.1f} mm")

    reasons.extend(
        workspace_violation_reasons(
            x_mm,
            y_mm,
            z_mm,
            settings,
        )
    )

    if validation.get("check_side_view_orientation", False):
        if wrist_r < 0.0:
            reasons.append("wrist target is not in front of the shoulder")

    if validation["check_joint_limits"]:
        for joint_name, angle in angles.items():
            joint = servo["joints"][joint_name]
            if not (joint["theta_min_deg"] <= angle <= joint["theta_max_deg"]):
                reasons.append(
                    f"{joint_name} angle {angle:.1f} deg outside "
                    f"{joint['theta_min_deg']:.1f} - {joint['theta_max_deg']:.1f} deg"
                )

    if validation["check_pulse_limits"]:
        for joint_name, angle in angles.items():
            joint = servo["joints"][joint_name]
            raw_pulse = angle_to_pwm_unclamped(angle, joint)
            if not (
                    joint["pulse_min_us"]
                    <= raw_pulse
                    <= joint["pulse_max_us"]
            ):
                reasons.append(
                    f"{joint_name} requires {raw_pulse:.0f} us outside "
                    f"{joint['pulse_min_us']:.0f} - "
                    f"{joint['pulse_max_us']:.0f} us"
                )

    return {
        "angles_deg": {
            "base": theta1,
            "shoulder": theta2,
            "elbow": theta3,
            "wrist": theta4,
        },
        "pwm_us": {
            "J1": angle_to_pwm(theta1, joint_1),
            "J2": angle_to_pwm(theta2, joint_2),
            "J3": angle_to_pwm(theta3, joint_3),
            "J4": angle_to_pwm(theta4, joint_4),
        },
        "reachable": not reasons,
        "reasons": reasons,
    }


def _print_result(x_mm: float, y_mm: float, z_mm: float) -> None:
    result = calculate_angles(x_mm, y_mm, z_mm)
    angles = result["angles_deg"]
    pwm = result["pwm_us"]

    print("Target position:")
    print(f"x = {x_mm:g} mm")
    print(f"y = {y_mm:g} mm")
    print(f"z = {z_mm:g} mm")
    print("")
    print("Calculated:")
    print(f"base angle = {angles['base']:.2f}°")
    print(f"shoulder angle = {angles['shoulder']:.2f}°")
    print(f"elbow angle = {angles['elbow']:.2f}°")
    print(f"wrist angle = {angles['wrist']:.2f}°")
    print("")
    print("PWM estimate:")
    print(f"J1 = {pwm['J1']} µs")
    print(f"J2 = {pwm['J2']} µs")
    print(f"J3 = {pwm['J3']} µs")
    print(f"J4 = {pwm['J4']} µs")
    print("")
    print(f"Reachable: {'yes' if result['reachable'] else 'no'}")
    if result["reasons"]:
        print("Reason:", "; ".join(result["reasons"]))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        _print_result(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    else:
        _print_result(200.0, 180.0, 60.0)
