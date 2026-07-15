import sys
import unittest
from math import hypot
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_machine.pick_and_place import MotionCommand, PickAndPlaceStateMachine, TargetPosition


class RecordingSink:
    def __init__(self) -> None:
        self.commands: list[MotionCommand] = []

    def send(self, command: MotionCommand) -> None:
        self.commands.append(command)


class PickAndPlaceRegressionTests(unittest.TestCase):
    def test_current_target_completes_dry_run_with_correct_joint_mapping(self) -> None:
        sink = RecordingSink()
        machine = PickAndPlaceStateMachine(sink)

        machine.start_pick_and_place(TargetPosition(230.0, 180.0, 60.0))
        success = machine.run_until_finished()

        self.assertTrue(success, machine.last_error)
        self.assertIsNone(machine.last_error)
        self.assertEqual(
            [command.name for command in sink.commands],
            [
                "move_ready",
                "move_in_front_of_object",
                "advance_towards_object",
                "close_gripper",
                "lift_object",
                "retract_from_shelf",
                "move_deposit",
                "open_gripper",
                "move_home",
            ],
        )
        advance = next(command for command in sink.commands if command.name == "advance_towards_object")
        self.assertEqual(
            advance.pulses_us,
            {
                "J1_base": 1608,
                "J2_shoulder": 1588,
                "J3_elbow": 1560,
                "J4_wrist": 1167,
            },
        )

        ready = next(command for command in sink.commands if command.name == "move_ready")
        self.assertAlmostEqual(ready.gripper_center_mm["x_mm"], 200.0, places=7)
        self.assertAlmostEqual(ready.gripper_center_mm["y_mm"], 120.0, places=7)
        self.assertAlmostEqual(ready.gripper_center_mm["z_mm"], 0.0, places=7)

        deposit = next(command for command in sink.commands if command.name == "move_deposit")
        self.assertIsNotNone(deposit.joint_angles_deg)
        self.assertAlmostEqual(deposit.gripper_center_mm["x_mm"], 0.0, places=7)
        self.assertAlmostEqual(deposit.gripper_center_mm["y_mm"], 370.0, places=7)
        self.assertAlmostEqual(deposit.gripper_center_mm["z_mm"], 130.0, places=7)
        self.assertEqual(deposit.pulses_us["J1_base"], 2167)
        self.assertEqual(deposit.pulses_us["J3_elbow"], 952)

    def test_intermediate_targets_follow_down_positive_y_and_radial_offsets(self) -> None:
        machine = PickAndPlaceStateMachine(RecordingSink())
        target = TargetPosition(230.0, 180.0, 60.0)
        machine.target = target

        in_front = machine._target_in_front_of_object()
        lifted = machine._target_lifted_from_object()
        retracted = machine._target_retracted_from_shelf()

        self.assertAlmostEqual(hypot(in_front.x_mm, in_front.z_mm), hypot(230.0, 60.0) - 20.0)
        self.assertEqual(in_front.y_mm, 140.0)
        self.assertEqual(lifted, TargetPosition(230.0, 130.0, 60.0))
        self.assertAlmostEqual(
            hypot(retracted.x_mm, retracted.z_mm),
            hypot(230.0, 60.0) - 20.0,
        )
        self.assertEqual(retracted.y_mm, 130.0)


if __name__ == "__main__":
    unittest.main()
