from math import inf, nan

import pytest

from planning.models import MotionPlan, PlanningFailure, TargetPose, ValidationStatus
from planning.pick_and_place_planner import PickAndPlacePlanner


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
    ).plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, MotionPlan)
    assert len(result.motions) == 9
    assert all(
        waypoint.validation_status is ValidationStatus.VALID
        for waypoint in result.waypoints
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
    result = PickAndPlacePlanner().plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, PlanningFailure)
    assert result.waypoint == "ready"
    assert result.code == "HARDWARE_SAFE_LIMIT_VIOLATION"
    assert "J2_shoulder" in result.message
    assert "1000-2000 us" in result.message


def test_invalid_named_pose_reports_joint_limit_code() -> None:
    planner = PickAndPlacePlanner(enforce_hardware_safe_limits=False)
    planner.poses["poses"]["ready"]["J2_shoulder"] = -200.0

    result = planner.plan(TargetPose(230.0, 180.0, 60.0))

    assert isinstance(result, PlanningFailure)
    assert result.waypoint == "ready"
    assert result.code == "JOINT_LIMIT_VIOLATION"
    assert result.rejected_waypoint.validation_status is ValidationStatus.INVALID
