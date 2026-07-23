from types import SimpleNamespace

import pytest

from config.config_loader import load_config
from kinematics.singularity_policy import (
    SingularityPolicy,
    singularity_policy_from_settings,
    singularity_policy_rejection_reasons,
)


def _analysis(
    *,
    geometric_status: str = "regular",
    conditioning_status: str = "regular",
    inverse_condition_number: float = 0.5,
    joint_limit_margin: float = 0.5,
    pulse_limit_margin: float = 0.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        geometric_status=geometric_status,
        conditioning_status=conditioning_status,
        metrics=SimpleNamespace(
            inverse_condition_number=inverse_condition_number
        ),
        joint_limit_margin=joint_limit_margin,
        pulse_limit_margin=pulse_limit_margin,
    )


def test_repository_singularity_policy_matches_planning_requirements() -> None:
    policy = singularity_policy_from_settings(
        load_config("kinematics_settings.toml")
    )

    assert policy == SingularityPolicy(
        reject_singular=True,
        reject_near_singular=False,
        minimum_inverse_condition_number=0.02,
        minimum_joint_limit_margin=0.05,
        minimum_pulse_limit_margin=0.05,
    )


def test_policy_reports_every_violated_threshold() -> None:
    policy = SingularityPolicy(
        reject_singular=True,
        reject_near_singular=True,
        minimum_inverse_condition_number=0.02,
        minimum_joint_limit_margin=0.05,
        minimum_pulse_limit_margin=0.05,
    )

    reasons = singularity_policy_rejection_reasons(
        _analysis(
            geometric_status="singular",
            conditioning_status="near_singular",
            inverse_condition_number=0.01,
            joint_limit_margin=0.04,
            pulse_limit_margin=0.03,
        ),
        policy,
    )

    assert len(reasons) == 5
    assert any("singular" in reason for reason in reasons)
    assert any("near-singular" in reason for reason in reasons)
    assert any("Inverse condition number" in reason for reason in reasons)
    assert any("Joint-limit margin" in reason for reason in reasons)
    assert any("Pulse-limit margin" in reason for reason in reasons)


def test_near_singular_classification_can_be_allowed() -> None:
    policy = SingularityPolicy(
        reject_singular=True,
        reject_near_singular=False,
        minimum_inverse_condition_number=0.0,
        minimum_joint_limit_margin=0.0,
        minimum_pulse_limit_margin=0.0,
    )

    reasons = singularity_policy_rejection_reasons(
        _analysis(
            conditioning_status="near_singular",
            inverse_condition_number=0.01,
        ),
        policy,
    )

    assert reasons == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reject_singular", 1),
        ("reject_near_singular", "false"),
        ("minimum_inverse_condition_number", -0.01),
        ("minimum_joint_limit_margin", 1.01),
        ("minimum_pulse_limit_margin", float("nan")),
    ],
)
def test_invalid_policy_configuration_is_rejected(
    field: str,
    value: object,
) -> None:
    settings = load_config("kinematics_settings.toml")
    settings["validation"]["singularity"][field] = value

    with pytest.raises((TypeError, ValueError)):
        singularity_policy_from_settings(settings)
