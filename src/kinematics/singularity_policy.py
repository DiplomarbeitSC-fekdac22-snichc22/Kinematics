from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.analysis_models import ConfigurationAnalysis


@dataclass(frozen=True)
class SingularityPolicy:
    reject_singular: bool
    reject_near_singular: bool
    minimum_inverse_condition_number: float
    minimum_joint_limit_margin: float
    minimum_pulse_limit_margin: float


def _required_boolean(
    config: Mapping[str, Any],
    field: str,
) -> bool:
    value = config[field]
    if not isinstance(value, bool):
        raise TypeError(f"validation.singularity.{field} must be boolean")
    return value


def _required_normalized_value(
    config: Mapping[str, Any],
    field: str,
) -> float:
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"validation.singularity.{field} must be numeric"
        )

    normalized = float(value)
    if not isfinite(normalized):
        raise ValueError(
            f"validation.singularity.{field} must be finite"
        )
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"validation.singularity.{field} must be between 0 and 1"
        )
    return normalized


def singularity_policy_from_settings(
    kinematics_settings: Mapping[str, Any],
) -> SingularityPolicy:
    """Parse and validate 'validation.singularity' settings."""
    validation = kinematics_settings["validation"]
    if not isinstance(validation, Mapping):
        raise TypeError("validation must be a table")

    config = validation["singularity"]
    if not isinstance(config, Mapping):
        raise TypeError("validation.singularity must be a table")

    return SingularityPolicy(
        reject_singular=_required_boolean(config, "reject_singular"),
        reject_near_singular=_required_boolean(
            config,
            "reject_near_singular",
        ),
        minimum_inverse_condition_number=_required_normalized_value(
            config,
            "minimum_inverse_condition_number",
        ),
        minimum_joint_limit_margin=_required_normalized_value(
            config,
            "minimum_joint_limit_margin",
        ),
        minimum_pulse_limit_margin=_required_normalized_value(
            config,
            "minimum_pulse_limit_margin",
        ),
    )


def load_singularity_policy(
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
) -> SingularityPolicy:
    """Load the configured planning policy from kinematics settings"""
    settings = load_config("kinematics_settings.toml", config_dir)
    return singularity_policy_from_settings(settings)


def singularity_policy_rejection_reasons(
    analysis: ConfigurationAnalysis,
    policy: SingularityPolicy,
) -> tuple[str, ...]:
    """Return every configured policy condition violated by an analysis"""
    reasons: list[str] = []

    if policy.reject_singular and analysis.geometric_status == "singular":
        reasons.append("Configuration is singular")

    if (
        policy.reject_near_singular
        and analysis.conditioning_status == "near_singular"
    ):
        reasons.append("Configuration is near-singular")

    inverse_condition_number = float(
        analysis.metrics.inverse_condition_number
    )
    if (
        inverse_condition_number
        < policy.minimum_inverse_condition_number
    ):
        reasons.append(
            "Inverse condition number "
            f"{inverse_condition_number:.6f} is below the configured "
            f"minimum {policy.minimum_inverse_condition_number:.6f}"
        )

    if analysis.joint_limit_margin < policy.minimum_joint_limit_margin:
        reasons.append(
            "Joint-limit margin "
            f"{analysis.joint_limit_margin:.6f} is below the configured "
            f"minimum {policy.minimum_joint_limit_margin:.6f}"
        )

    if analysis.pulse_limit_margin < policy.minimum_pulse_limit_margin:
        reasons.append(
            "Pulse-limit margin "
            f"{analysis.pulse_limit_margin:.6f} is below the configured "
            f"minimum {policy.minimum_pulse_limit_margin:.6f}"
        )

    return tuple(reasons)
