from math import radians, cos, sin, pi
from pathlib import Path
from typing import Any

import numpy as np

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.analysis_models import JacobianResult


def _angle_for_role_rad(
        joint_angles_deg: dict[str, float],
        servo_config: dict[str, Any],
        role: str
) -> float:
    """Read a configured joint angle and convert degrees to radians."""
    for joint_name, joint in servo_config["joints"].items():
        if joint.get("kinematic_role") == role:
            if joint_name not in joint_angles_deg:
                raise KeyError(f"Missing angle for joint {joint_name!r}")
            return radians(float(joint_angles_deg[joint_name]))
    raise KeyError(f"No servo joint configured for role {role!r}")


def calculate_jacobian(
        joint_angles_deg: dict[str, float],
        config_dir: Path | str = DEFAULT_CONFIG_DIR
) -> JacobianResult:
    """Calculate the analytical and task-scaled 4x4 Jacobians.

    The unscaled Jacobian maps joint angular velocity in rad/s to
    [x_dot, y_dot, z_dot, alpha_dot], where position is measured in mm.
    """
    config_dir = Path(config_dir)
    geometry = load_config("robot_geometry.toml", config_dir)
    servo = load_config("servo_calibration.toml", config_dir)
    settings = load_config("kinematics_settings.toml", config_dir)
    analysis = load_config("singularity_analysis.toml", config_dir)

    links = geometry["link_lengths_mm"]
    model = settings["model"]

    l1 = float(links["L1_shoulder_to_elbow"])
    l2 = float(links["L2_elbow_to_wrist"])
    lg = (
        float(links[model["selected_Lg_key"]])
        if model["use_gripper_offset"]
        else 0.0
    )

    theta1 = _angle_for_role_rad(joint_angles_deg, servo, "theta1")
    theta2 = _angle_for_role_rad(joint_angles_deg, servo, "theta2")
    theta3 = _angle_for_role_rad(joint_angles_deg, servo, "theta3")
    theta4 = _angle_for_role_rad(joint_angles_deg, servo, "theta4")

    # Calculations
    elbow_relative_sign = float(settings.get("fk", {}).get("elbow_relative_sign", -1.0))
    delta = elbow_relative_sign * (pi - theta3)

    forearm_angle = theta2 + delta
    alpha = forearm_angle + theta4

    # rho = signed radial distance from the base axis to the gripper center
    rho = l1 * cos(theta2) + l2 * cos(forearm_angle) + lg * cos(alpha)

    # d(rho)/d(theta2): differentiating cos(u) gives -sin(u) * du/dtheta2
    drho_dtheta2 = -l1 * sin(theta2) - l2 * sin(forearm_angle) - lg * sin(alpha)

    # d(rho)/d(delta): L1 does not depend on delta; L2 and Lg do
    drho_ddelta = -l2 * sin(forearm_angle) - lg * sin(alpha)

    # d(rho)/d(theta4): only gripper-offset term depends on theta4
    drho_dtheta4 = -lg * sin(alpha)

    # v is the gripper height in a mathematical coordinate system with positive upward
    # Robot coordinate system points downward -> dy/dq = -dv/dq
    dv_dtheta2 = (
            l1 * cos(theta2)
            + l2 * cos(forearm_angle)
            + lg * cos(alpha)
    )

    # d(v)/d(delta): differentiate the L2 and Lg sine terms into cosine terms
    dv_ddelta = l2 * cos(forearm_angle) + lg * cos(alpha)

    # d(v)/d(theta4): only the gripper-offset sine term depends on wrist rotation
    dv_dtheta4 = lg * cos(alpha)

    # Create the matrix
    # It first uses delta as the third generalized coordinate
    # Row 1 differentiates x = rho*cos(theta1)
    # Row 2 differentiates public y = shoulder_y - v
    # Row 3 differentiates z = rho*sin(theta1)
    # Row 4 differentiates alpha = theta2 + delta + theta4
    jacobian_delta = np.array(
        [
            [
                -rho * sin(theta1),
                drho_dtheta2 * cos(theta1),
                drho_ddelta * cos(theta1),
                drho_dtheta4 * cos(theta1)
            ],
            [0.0, -dv_dtheta2, -dv_ddelta, -dv_dtheta4],
            [
                rho * cos(theta1),
                drho_dtheta2 * sin(theta1),
                drho_ddelta * sin(theta1),
                drho_dtheta4 * sin(theta1)
            ],
            [0.0, 1.0, 1.0, 1.0]
        ],
        dtype=float
    )

    # delta = s*(pi-theta3), therefore d(delta)/d(theta3) = -s
    # Column 3 * -s: Jacobian -> repository joint coordinates
    jacobian = jacobian_delta.copy()
    jacobian[:, 2] *= -elbow_relative_sign

    source = analysis["task"]["characteristic_length_source"]
    if source != "gripper_offset":
        raise ValueError(f"Unsupported characteristic_length_source: {source!r}")

    characteristic_length = lg
    scaling = np.diag([1.0, 1.0, 1.0, characteristic_length])
    scaled_jacobian = scaling @ jacobian

    return JacobianResult(
        jacobian=jacobian,
        scaled_jacobian=scaled_jacobian,
        radial_distance_mm=rho,
        elbow_relative_angle_rad=delta,
        approach_angle_rad=alpha,
        characteristic_length_mm=characteristic_length
    )
