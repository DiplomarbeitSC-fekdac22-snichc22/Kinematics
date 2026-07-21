from math import sin
from pathlib import Path

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.analysis_models import ConfigurationAnalysis
from kinematics.jacobian import calculate_jacobian
from kinematics.manipulability import calculate_manipulability_metrics


def analyze_configuration(
        joint_angles_deg: dict[str, float],
        config_dir: Path | str = DEFAULT_CONFIG_DIR
) -> ConfigurationAnalysis:
    """Analyze one J1-J4 configuration without moving hardware."""
    config_dir = Path(config_dir)
    geometry = load_config("robot_geometry.toml", config_dir)
    servo = load_config("servo_calibration.toml", config_dir)
    analysis_config = load_config("singularity_analysis.toml", config_dir)

    jacobian_result = calculate_jacobian(joint_angles_deg, config_dir)

    links = geometry["link_lengths_mm"]
    l1 = float(links["L1_shoulder_to_elbow"])
    l2 = float(links["L2_elbow_to_wrist"])
    lg = jacobian_result.characteristic_length_mm

    maximum_radial_reach = l1 + l2 + lg

    numerical = analysis_config["numerics"]
    classification = analysis_config["classification"]

    metrics = calculate_manipulability_metrics(
        scaled_jacobian=jacobian_result.scaled_jacobian,
        radial_distance_mm=jacobian_result.radial_distance_mm,
        maximum_radial_reach_mm=maximum_radial_reach,
        rank_relative_tolerance=float(numerical["rank_relative_tolerance"]),
    )

    exact_tolerance = float(numerical["exact_singularity_tolerance"])

    base_axis_singularity = abs(jacobian_result.radial_distance_mm) <= exact_tolerance
    elbow_singularity = abs(sin(jacobian_result.elbow_relative_angle_rad)) <= exact_tolerance

    geometric_status = (
        "singular"
        if metrics.rank < 4 or base_axis_singularity or elbow_singularity
        else "regular"
    )

    severe_threshold = float(classification["inverse_condition_severe"])
    warning_threshold = float(classification["inverse_condition_warning"])

    if geometric_status == "singular":
        condition_status = "singular"
    elif metrics.inverse_condition_number < severe_threshold:
        condition_status = "near_singular"
    elif metrics.inverse_condition_number < warning_threshold:
        condition_status = "warning"
    else:
        condition_status = "regular"

    joint_margins: list[float] = []
    pulse_margins: list[float] = []
    warnings: list[str] = []

