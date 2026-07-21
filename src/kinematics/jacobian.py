from math import radians
from pathlib import Path
from typing import Any

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
