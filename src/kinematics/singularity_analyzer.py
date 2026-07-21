from math import sin
from pathlib import Path
from typing import Any

from kinematics.angle_to_pwm import angle_to_pwm_unclamped

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.analysis_models import ConfigurationAnalysis
from kinematics.jacobian import calculate_jacobian
from kinematics.manipulability import calculate_manipulability_metrics


def _joint_name_for_role(servo_config: dict[str, Any], role: str) -> str:
    for joint_name, joint in servo_config["joints"].items():
        if joint.get("kinematic_role") == role:
            return joint_name
    raise KeyError(f"No servo joint configured for role {role!r}")


def _normalized_interval_margin(value: float, minimum: float, maximum: float) -> float:
    """Return 1 at interval centre, 0 at an endpoint, and negative outside."""
    if maximum <= minimum:
        raise ValueError("Interval max must be greater than min")

    nearest_distance = min(value - minimum, maximum - value)

    return 2.0 * nearest_distance / (maximum - minimum)


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
        conditioning_status = "singular"
    elif metrics.inverse_condition_number < severe_threshold:
        conditioning_status = "near_singular"
    elif metrics.inverse_condition_number < warning_threshold:
        conditioning_status = "warning"
    else:
        conditioning_status = "regular"

    joint_margins: list[float] = []
    pulse_margins: list[float] = []
    warnings: list[str] = []

    for role in ("theta1", "theta2", "theta3", "theta4"):
        joint_name = _joint_name_for_role(servo, role)
        joint = servo["joints"][joint_name]
        angle = float(joint_angles_deg[joint_name])

        joint_margin = _normalized_interval_margin(
            angle,
            float(joint["theta_min_deg"]),
            float(joint["theta_max_deg"]),
        )
        joint_margins.append(joint_margin)

        raw_pulse = angle_to_pwm_unclamped(angle, joint)
        pulse_margin = _normalized_interval_margin(
            raw_pulse,
            float(joint["pulse_min_us"]),
            float(joint["pulse_max_us"]),
        )
        pulse_margins.append(pulse_margin)

        if joint_margin < 0.0:
            warnings.append(f"{joint_name} is outside its angle limits")
        if pulse_margin < 0.0:
            warnings.append(f"{joint_name} requires a pulse outside its limits")

    joint_limit_margin = min(joint_margins)
    pulse_limit_margin = min(pulse_margins)

    joint_warning = float(classification["joint_limit_warning_margin"])
    pulse_warning = float(classification["pulse_limit_warning_margin"])

    if joint_limit_margin < 0.0 or pulse_limit_margin < 0.0:
        constraint_status = "invalid"
    elif joint_limit_margin < joint_warning or pulse_limit_margin < pulse_warning:
        constraint_status = "warning"
    else:
        constraint_status = "regular"

    if base_axis_singularity:
        warnings.append("Gripper centre lies on the base rotation axis")
    if elbow_singularity:
        warnings.append("Upper arm and forearm are collinear")
    if conditioning_status == "near_singular":
        warnings.append("Configuration is severely ill-conditioned")
    elif conditioning_status == "warning":
        warnings.append("Configuration is approaching a singular region")

    return ConfigurationAnalysis(
        joint_angles_deg=dict(joint_angles_deg),
        jacobian=jacobian_result,
        metrics=metrics,
        base_axis_singularity=base_axis_singularity,
        elbow_singularity=elbow_singularity,
        geometric_status=geometric_status,
        conditioning_status=conditioning_status,
        joint_limit_margin=joint_limit_margin,
        pulse_limit_margin=pulse_limit_margin,
        constraint_status=constraint_status,
        warnings=tuple(warnings),
    )
