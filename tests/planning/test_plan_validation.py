from math import inf, nan

import pytest

import planning.pick_and_place_planner as planner_module
from planning.models import MotionPlan, PlanningFailure, TargetPose, ValidationStatus
from planning.pick_and_place_planner import PickAndPlacePlanner
from tests.policy_helpers import PERMISSIVE_SINGULARITY_POLICY


@pytest.mark.parametrize("bad_value", [nan, inf, -inf])
def test_rejects_non_finite_cartesian_values(bad_value: float) -> None:
    result = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
    ).plan(TargetPose(bad_value, 180.0, 60.0))

    assert isinstance(result, PlanningFailure)
    assert result.waypoint == "grasp"
    assert result.code == "INVALID_CARTESIAN_TARGET"


def test_all_waypoints_contain_stored_validation_results() -> None:
    result = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
        singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
    ).plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, MotionPlan)
    assert len(result.motions) == 9
    assert all(
        waypoint.validation_status is ValidationStatus.VALID
        for waypoint in result.waypoints
    )
    assert all(
        waypoint.singularity_analysis is not None
        for waypoint in result.waypoints
    )
    assert (
        result.motion_for("close_gripper").waypoint.singularity_analysis
        is result.motion_for("grasp").waypoint.singularity_analysis
    )
    assert (
        result.motion_for("open_gripper").waypoint.singularity_analysis
        is result.motion_for("deposit").waypoint.singularity_analysis
    )
    assert result.motion_for("grasp").waypoint.ik_branch == "elbow_back"
    assert result.motion_for("grasp").command.pulses_us == {
        "J1_base": 1608,
        "J2_shoulder": 777,
        "J3_elbow": 1208,
        "J4_wrist": 970,
    }
    assert any("effective hardware-safe range" in warning for warning in result.warnings)


def test_strict_hardware_preflight_rejects_current_provisional_pose() -> None:
    result = PickAndPlacePlanner(
        singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
    ).plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, PlanningFailure)
    assert result.waypoint == "ready"
    assert result.code == "HARDWARE_SAFE_LIMIT_VIOLATION"
    assert "J2_shoulder" in result.message
    assert "1000-2000 us" in result.message


def test_invalid_named_pose_reports_joint_limit_code() -> None:
    planner = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
        singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
    )
    planner.poses["poses"]["ready"]["J2_shoulder"] = -200.0

    result = planner.plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, PlanningFailure)
    assert result.waypoint == "ready"
    assert result.code == "JOINT_LIMIT_VIOLATION"
    assert result.rejected_waypoint.validation_status is ValidationStatus.INVALID


def test_each_cartesian_waypoint_uses_previous_selected_joint_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_states: list[dict[str, float]] = []
    real_selector = planner_module.select_continuous_solution

    def recording_selector(
        solutions,
        current_joint_angles,
        config_dir,
        *,
        policy,
    ):
        received_states.append(dict(current_joint_angles))
        return real_selector(
            solutions,
            current_joint_angles,
            config_dir,
            policy=policy,
        )

    monkeypatch.setattr(
        planner_module,
        "select_continuous_solution",
        recording_selector,
    )
    planner = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
        singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
    )

    result = planner.plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, MotionPlan)
    selected_cartesian_states = [
        result.motion_for(name).command.joint_angles_deg
        for name in ("pre_grasp", "grasp", "lift", "retract")
    ]
    assert received_states[0] == {
        key: float(value)
        for key, value in planner.poses["poses"]["ready"].items()
        if key.startswith("J")
    }
    assert received_states[1:] == selected_cartesian_states


def test_default_policy_rejects_provisional_ready_pose_at_limit() -> None:
    result = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
    ).plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, PlanningFailure)
    assert result.waypoint == "ready"
    assert result.code == "SINGULARITY_POLICY_VIOLATION"
    assert result.rejected_waypoint is not None
    assert result.rejected_waypoint.singularity_analysis is not None
    assert any("Joint-limit margin" in reason for reason in result.reasons)
    assert any("Pulse-limit margin" in reason for reason in result.reasons)
