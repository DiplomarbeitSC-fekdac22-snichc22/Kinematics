from math import hypot

import pytest

from planning.models import TargetPose
from planning.waypoint_generator import WaypointGenerator


def test_generates_complete_sequence_in_execution_order() -> None:
    generator = WaypointGenerator()

    waypoints = generator.generate(TargetPose(230.0, 180.0, 60.0))

    assert [waypoint.name for waypoint in waypoints] == [
        "ready",
        "pre_grasp",
        "grasp",
        "close_gripper",
        "lift",
        "retract",
        "deposit",
        "open_gripper",
        "home",
    ]
    assert [waypoint.command_name for waypoint in waypoints] == [
        "move_ready",
        "move_in_front_of_object",
        "advance_towards_object",
        "close_gripper",
        "lift_object",
        "retract_from_shelf",
        "move_deposit",
        "open_gripper",
        "move_home",
    ]


def test_cartesian_offsets_are_generated_outside_the_state_machine() -> None:
    generator = WaypointGenerator()
    target = TargetPose(230.0, 180.0, 60.0)
    generated = {item.name: item for item in generator.generate(target)}
    offset = float(
        generator.kinematics_settings["target_offsets"]["approach_r_offset_mm"]
    )

    assert generated["grasp"].cartesian_target == target
    assert generated["lift"].cartesian_target == TargetPose(230.0, 130.0, 60.0)
    assert generated["pre_grasp"].cartesian_target.y_mm == 140.0
    assert generated["retract"].cartesian_target.y_mm == 130.0
    assert hypot(
        generated["pre_grasp"].cartesian_target.x_mm,
        generated["pre_grasp"].cartesian_target.z_mm,
    ) == pytest.approx(hypot(230.0, 60.0) - offset)
    assert hypot(
        generated["retract"].cartesian_target.x_mm,
        generated["retract"].cartesian_target.z_mm,
    ) == pytest.approx(hypot(230.0, 60.0) - offset)
