from types import SimpleNamespace

import pytest

import kinematics.solution_selector as selector


CURRENT_JOINT_STATE = {
    "J1_base": 0.0,
    "J2_shoulder": 0.0,
    "J3_elbow": 90.0,
    "J4_wrist": 0.0,
}


def _solution(
    branch: str,
    shoulder_deg: float,
    *,
    reachable: bool = True,
) -> dict[str, object]:
    return {
        "branch": branch,
        "elbow_relative_sign": (
            1.0 if branch == "elbow_back" else -1.0
        ),
        "angles_deg": {
            "base": 0.0,
            "shoulder": shoulder_deg,
            "elbow": 90.0,
            "wrist": 0.0,
        },
        "reachable": reachable,
    }


def _install_analysis_stub(
    monkeypatch: pytest.MonkeyPatch,
    values_by_shoulder: dict[float, dict[str, object]],
) -> None:
    def analyze(joint_angles, _config_dir, *, elbow_relative_sign):
        values = values_by_shoulder[float(joint_angles["J2_shoulder"])]
        return SimpleNamespace(
            geometric_status=values.get("geometric_status", "regular"),
            constraint_status=values.get("constraint_status", "regular"),
            joint_limit_margin=values.get("joint_limit_margin", 0.5),
            pulse_limit_margin=values.get("pulse_limit_margin", 0.5),
            metrics=SimpleNamespace(
                rank=values.get("rank", 4),
                inverse_condition_number=values.get(
                    "inverse_condition_number",
                    0.5,
                ),
            ),
        )

    monkeypatch.setattr(selector, "analyze_configuration", analyze)


def test_valid_solution_beats_closer_invalid_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_stub(monkeypatch, {0.0: {}, 10.0: {}})
    invalid_close = _solution("elbow_back", 0.0, reachable=False)
    valid_far = _solution("elbow_forward", 10.0)

    selected = selector.select_continuous_solution(
        [invalid_close, valid_far],
        CURRENT_JOINT_STATE,
    )

    assert selected.solution is valid_far
    assert selected.valid


def test_distance_beats_later_quality_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_stub(
        monkeypatch,
        {
            1.0: {
                "geometric_status": "singular",
                "rank": 3,
                "inverse_condition_number": 0.0,
                "joint_limit_margin": 0.1,
                "pulse_limit_margin": 0.1,
            },
            -20.0: {
                "inverse_condition_number": 1.0,
                "joint_limit_margin": 1.0,
                "pulse_limit_margin": 1.0,
            },
        },
    )
    close = _solution("elbow_forward", 1.0)
    far = _solution("elbow_back", -20.0)

    selected = selector.select_continuous_solution(
        [far, close],
        CURRENT_JOINT_STATE,
    )

    assert selected.solution is close
    assert selected.joint_distance_deg == pytest.approx(1.0)


def test_singularity_quality_beats_limit_margins_and_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_stub(
        monkeypatch,
        {
            10.0: {
                "geometric_status": "singular",
                "rank": 3,
                "inverse_condition_number": 0.0,
                "joint_limit_margin": 1.0,
                "pulse_limit_margin": 1.0,
            },
            -10.0: {
                "inverse_condition_number": 0.2,
                "joint_limit_margin": 0.1,
                "pulse_limit_margin": 0.1,
            },
        },
    )
    preferred_singular = _solution("elbow_back", 10.0)
    regular = _solution("elbow_forward", -10.0)

    selected = selector.select_continuous_solution(
        [preferred_singular, regular],
        CURRENT_JOINT_STATE,
    )

    assert selected.solution is regular


def test_joint_margin_beats_pulse_margin_and_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_stub(
        monkeypatch,
        {
            10.0: {
                "joint_limit_margin": 0.2,
                "pulse_limit_margin": 1.0,
            },
            -10.0: {
                "joint_limit_margin": 0.8,
                "pulse_limit_margin": 0.1,
            },
        },
    )
    preferred_low_joint_margin = _solution("elbow_back", 10.0)
    high_joint_margin = _solution("elbow_forward", -10.0)

    selected = selector.select_continuous_solution(
        [preferred_low_joint_margin, high_joint_margin],
        CURRENT_JOINT_STATE,
    )

    assert selected.solution is high_joint_margin


def test_pulse_margin_beats_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_stub(
        monkeypatch,
        {
            10.0: {"pulse_limit_margin": 0.2},
            -10.0: {"pulse_limit_margin": 0.8},
        },
    )
    preferred_low_pulse_margin = _solution("elbow_back", 10.0)
    high_pulse_margin = _solution("elbow_forward", -10.0)

    selected = selector.select_continuous_solution(
        [preferred_low_pulse_margin, high_pulse_margin],
        CURRENT_JOINT_STATE,
    )

    assert selected.solution is high_pulse_margin


def test_configured_branch_preference_breaks_exact_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_analysis_stub(monkeypatch, {10.0: {}, -10.0: {}})
    preferred = _solution("elbow_back", 10.0)
    other = _solution("elbow_forward", -10.0)

    selected = selector.select_continuous_solution(
        [other, preferred],
        CURRENT_JOINT_STATE,
    )

    assert selected.solution is preferred
