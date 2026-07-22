from collections import defaultdict
from math import sqrt
from statistics import fmean
from typing import Any, Iterable, Mapping

from calibration.models import JointCalibrationFit, PulseAngleMeasurement


DEFAULT_MINIMUM_PAIRS = 3


class CalibrationDataError(ValueError):
    """Raised when measured data cannot produce a trustworthy linear fit."""


def fit_joint_calibration(
    joint_name: str,
    measurements: Iterable[PulseAngleMeasurement],
    *,
    theta_zero_deg: float,
    minimum_pairs: int = DEFAULT_MINIMUM_PAIRS,
) -> JointCalibrationFit:
    """Fit one joint from several measured pulse/angle pairs."""
    if minimum_pairs < 2:
        raise ValueError("minimum_pairs must be at least 2")

    samples = list(measurements)
    if any(sample.joint_name != joint_name for sample in samples):
        raise CalibrationDataError(
            f"Measurements for {joint_name} contain a different joint name"
        )
    if len(samples) < minimum_pairs:
        raise CalibrationDataError(
            f"{joint_name} requires at least {minimum_pairs} measured pairs; "
            f"got {len(samples)}"
        )

    unique_angles = {sample.angle_deg for sample in samples}
    if len(unique_angles) < minimum_pairs:
        raise CalibrationDataError(
            f"{joint_name} requires at least {minimum_pairs} distinct measured "
            f"angles; got {len(unique_angles)}"
        )

    angles = [sample.angle_deg for sample in samples]
    pulses = [sample.pulse_us for sample in samples]
    mean_angle = fmean(angles)
    mean_pulse = fmean(pulses)
    angle_variance_sum = sum((angle - mean_angle) ** 2 for angle in angles)
    if angle_variance_sum == 0.0:
        raise CalibrationDataError(f"{joint_name} measured angles have no span")

    slope = sum(
        (angle - mean_angle) * (pulse - mean_pulse)
        for angle, pulse in zip(angles, pulses, strict=True)
    ) / angle_variance_sum
    if slope == 0.0:
        raise CalibrationDataError(
            f"{joint_name} measurements produce a zero pulse/angle slope"
        )

    intercept = mean_pulse - slope * mean_angle
    predictions = [intercept + slope * angle for angle in angles]
    residuals = [
        pulse - predicted
        for pulse, predicted in zip(pulses, predictions, strict=True)
    ]
    squared_error_sum = sum(residual**2 for residual in residuals)
    total_variation = sum((pulse - mean_pulse) ** 2 for pulse in pulses)
    if total_variation == 0.0:
        raise CalibrationDataError(
            f"{joint_name} measured pulses have no usable variation"
        )

    return JointCalibrationFit(
        joint_name=joint_name,
        sample_count=len(samples),
        unique_angle_count=len(unique_angles),
        theta_zero_deg=theta_zero_deg,
        slope_us_per_degree=slope,
        intercept_us=intercept,
        direction=1 if slope > 0.0 else -1,
        us_per_degree=abs(slope),
        pulse_center_us=intercept + slope * theta_zero_deg,
        observed_angle_min_deg=min(angles),
        observed_angle_max_deg=max(angles),
        observed_pulse_min_us=min(pulses),
        observed_pulse_max_us=max(pulses),
        rmse_us=sqrt(squared_error_sum / len(residuals)),
        mean_absolute_error_us=fmean(abs(residual) for residual in residuals),
        maximum_absolute_error_us=max(abs(residual) for residual in residuals),
        r_squared=1.0 - squared_error_sum / total_variation,
    )


def fit_servo_calibrations(
    measurements: Iterable[PulseAngleMeasurement],
    servo_calibration: Mapping[str, Any],
    *,
    minimum_pairs: int = DEFAULT_MINIMUM_PAIRS,
) -> dict[str, JointCalibrationFit]:
    """Fit all joints present in a measurement set using configured zeros."""
    groups: dict[str, list[PulseAngleMeasurement]] = defaultdict(list)
    for measurement in measurements:
        groups[measurement.joint_name].append(measurement)

    if not groups:
        raise CalibrationDataError("No pulse/angle measurements were provided")

    configured_joints = servo_calibration.get("joints")
    if not isinstance(configured_joints, Mapping):
        raise CalibrationDataError(
            "Servo calibration is missing the joints table"
        )

    unknown_joints = sorted(set(groups) - set(configured_joints))
    if unknown_joints:
        raise CalibrationDataError(
            f"Measurements contain unknown joints: {tuple(unknown_joints)}"
        )

    fits: dict[str, JointCalibrationFit] = {}
    for joint_name in sorted(groups):
        joint = configured_joints[joint_name]
        if not isinstance(joint, Mapping) or "theta_zero_deg" not in joint:
            raise CalibrationDataError(
                f"{joint_name} is missing configured theta_zero_deg"
            )
        fits[joint_name] = fit_joint_calibration(
            joint_name,
            groups[joint_name],
            theta_zero_deg=float(joint["theta_zero_deg"]),
            minimum_pairs=minimum_pairs,
        )

    return fits
