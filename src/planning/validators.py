from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping

from kinematics.angle_to_pwm import (
    AngleOutsideCalibrationError,
    PulseLimitError,
    angle_to_pwm,
)
from kinematics.workspace_checker import workspace_violation_reasons
from planning.models import TargetPose


@dataclass(frozen=True)
class EffectivePulseRange:
    minimum_us: int
    maximum_us: int


def validate_xyz_values(target: TargetPose) -> tuple[str, ...]:
    """Reject non-numeric and non-finite Cartesian coordinates."""
    reasons: list[str] = []
    for name, value in (
        ("x_mm", target.x_mm),
        ("y_mm", target.y_mm),
        ("z_mm", target.z_mm),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            reasons.append(f"{name} must be numeric")
        elif not isfinite(float(value)):
            reasons.append(f"{name} must be finite")
    return tuple(reasons)


def validate_workspace(
    target: TargetPose,
    kinematics_settings: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        workspace_violation_reasons(
            target.x_mm,
            target.y_mm,
            target.z_mm,
            kinematics_settings,
        )
    )


def validate_joint_angles(
    joint_angles_deg: Mapping[str, float],
    servo_calibration: dict[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    joints = servo_calibration["joints"]

    for joint_name, angle in joint_angles_deg.items():
        if joint_name not in joints:
            reasons.append(f"Unknown joint {joint_name}")
            continue
        if isinstance(angle, bool) or not isinstance(angle, Real):
            reasons.append(f"{joint_name} angle must be numeric")
            continue
        if not isfinite(float(angle)):
            reasons.append(f"{joint_name} angle must be finite")
            continue

        joint = joints[joint_name]
        minimum = float(joint["theta_min_deg"])
        maximum = float(joint["theta_max_deg"])
        if not minimum <= float(angle) <= maximum:
            reasons.append(
                f"{joint_name} angle {float(angle):.1f} deg outside "
                f"{minimum:.1f} - {maximum:.1f} deg"
            )

    return tuple(reasons)


def convert_joint_angles_to_pwm(
    joint_angles_deg: Mapping[str, float],
    servo_calibration: dict[str, Any],
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Strictly convert command angles and report calibration violations."""
    pulses: dict[str, int] = {}
    reasons: list[str] = []
    joints = servo_calibration["joints"]

    for joint_name, angle in joint_angles_deg.items():
        if joint_name not in joints:
            reasons.append(f"Unknown joint {joint_name}")
            continue

        joint = joints[joint_name]
        try:
            pulse_us = angle_to_pwm(float(angle), joint)
        except AngleOutsideCalibrationError as exc:
            reasons.append(
                f"{joint_name} angle {exc.angle_deg:.1f} deg outside "
                f"{exc.minimum_deg:.1f} - {exc.maximum_deg:.1f} deg"
            )
            continue
        except PulseLimitError as exc:
            reasons.append(
                f"{joint_name} requires {exc.pulse_us} us outside "
                f"{exc.minimum_us} - {exc.maximum_us} us"
            )
            continue

        pulses[joint_name] = pulse_us

    return pulses, tuple(reasons)


def effective_hardware_pulse_range(
    joint: dict[str, Any],
    defaults: dict[str, Any],
) -> EffectivePulseRange:
    minimum = max(
        int(joint["pulse_min_us"]),
        int(defaults["pulse_electrical_min_us"]),
    )
    maximum = min(
        int(joint["pulse_max_us"]),
        int(defaults["pulse_electrical_max_us"]),
    )

    if bool(defaults.get("clamp_to_initial_safe_range", False)):
        minimum = max(
            minimum,
            int(defaults["pulse_initial_safe_min_us"]),
        )
        maximum = min(
            maximum,
            int(defaults["pulse_initial_safe_max_us"]),
        )

    if minimum > maximum:
        raise ValueError("Configured pulse ranges do not overlap")

    return EffectivePulseRange(minimum, maximum)


def validate_hardware_safe_pulses(
    pulses_us: Mapping[str, int],
    servo_calibration: dict[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    joints = servo_calibration["joints"]
    defaults = servo_calibration["defaults"]

    for joint_name, pulse_us in pulses_us.items():
        if joint_name not in joints:
            reasons.append(f"Unknown joint {joint_name}")
            continue

        allowed = effective_hardware_pulse_range(
            joints[joint_name],
            defaults,
        )
        if not allowed.minimum_us <= pulse_us <= allowed.maximum_us:
            reasons.append(
                f"{joint_name} requests {pulse_us} us; effective hardware-safe "
                f"range is {allowed.minimum_us}-{allowed.maximum_us} us"
            )

    return tuple(reasons)


def validate_gripper_pulse(
    joint_name: str,
    pulse_us: int,
    servo_calibration: dict[str, Any],
) -> tuple[str, ...]:
    joint = servo_calibration["joints"].get(joint_name)
    if joint is None:
        return (f"Unknown gripper joint {joint_name}",)

    minimum = int(joint["pulse_min_us"])
    maximum = int(joint["pulse_max_us"])
    if not minimum <= pulse_us <= maximum:
        return (
            f"{joint_name} command {pulse_us} us outside "
            f"{minimum}-{maximum} us",
        )
    return ()
