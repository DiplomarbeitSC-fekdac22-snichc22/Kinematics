from config.config_loader import load_config


def is_target_reachable(x_mm: float, y_mm: float, z_mm: float) -> bool:
    kinematics = load_config("kinematics_settings.toml")
    bounds = kinematics["workspace_bounds_robot_base_mm"]

    return (
        bounds["x_min"] <= x_mm <= bounds["x_max"]
        and bounds["y_min"] <= y_mm <= bounds["y_max"]
        and bounds["z_min"] <= z_mm <= bounds["z_max"]
    )