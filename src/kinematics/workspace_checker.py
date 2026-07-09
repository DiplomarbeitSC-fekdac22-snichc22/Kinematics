from config.config_loader import load_config


def is_target_inside_workspace_bounds(x_mm: float, y_mm: float, z_mm: float) -> bool:
    kinematics = load_config("kinematics_settings.toml")
    bounds = kinematics["workspace_bounds_robot_base_mm"]

    if not kinematics["validation"]["check_workspace_bounds"]:
        return True

    return (
        bounds["x_min"] <= x_mm <= bounds["x_max"]
        and bounds["y_min"] <= y_mm <= bounds["y_max"]
        and bounds["z_min"] <= z_mm <= bounds["z_max"]
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