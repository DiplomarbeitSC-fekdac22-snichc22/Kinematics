import sys
import unittest
from math import hypot
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api import RobotController
from config.config_loader import load_config
from kinematics.forward_kinematics import calculate_gripper_center
from planning.models import MotionCommand, TargetPose
from planning.waypoint_generator import WaypointGenerator


class RecordingSink:
    def __init__(self) -> None:
        self.commands: list[MotionCommand] = []

    def send(self, command: MotionCommand) -> None:
        self.commands.append(command)


class PickAndPlaceRegressionTests(unittest.TestCase):
    def test_current_target_completes_dry_run_with_correct_joint_mapping(self) -> None:
        sink = RecordingSink()
        controller = RobotController(sink)

        success = controller.run_pick_and_place(230.0, 180.0, 60.0)

        self.assertTrue(success)
        self.assertIsNotNone(controller.machine)
        self.assertIsNone(controller.machine.last_error)
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
        advance = next(
            command
            for command in sink.commands
            if command.name == "advance_towards_object"
        )
        self.assertEqual(
            advance.pulses_us,
            {
                "J1_base": 1608,
                "J2_shoulder": 777,
                "J3_elbow": 1208,
                "J4_wrist": 970,
            },
        )

        ready = next(command for command in sink.commands if command.name == "move_ready")
        ready_angles = {
            key: float(value)
            for key, value in load_config("poses.toml")["poses"]["ready"].items()
            if key.startswith("J")
        }
        expected_ready_center = calculate_gripper_center(ready_angles)
        self.assertEqual(ready.gripper_center_mm, expected_ready_center)
        self.assertEqual(ready.pulses_us["J5_gripper"], 1200)

        home = next(command for command in sink.commands if command.name == "move_home")
        self.assertEqual(home.pulses_us["J5_gripper"], 1200)

        deposit = next(command for command in sink.commands if command.name == "move_deposit")
        self.assertIsNotNone(deposit.joint_angles_deg)
        self.assertAlmostEqual(deposit.gripper_center_mm["x_mm"], 0.0, places=7)
        self.assertAlmostEqual(deposit.gripper_center_mm["y_mm"], 370.0, places=7)
        self.assertAlmostEqual(deposit.gripper_center_mm["z_mm"], 130.0, places=7)
        self.assertEqual(deposit.pulses_us["J1_base"], 2167)
        self.assertEqual(deposit.pulses_us["J3_elbow"], 1660)

    def test_intermediate_targets_follow_down_positive_y_and_radial_offsets(self) -> None:
        generator = WaypointGenerator()
        target = TargetPose(230.0, 180.0, 60.0)
        offsets = generator.kinematics_settings["target_offsets"]

        in_front = generator.target_in_front_of_object(target)
        lifted = generator.target_lifted_from_object(target)
        retracted = generator.target_retracted_from_shelf(target)

        self.assertAlmostEqual(
            hypot(in_front.x_mm, in_front.z_mm),
            hypot(230.0, 60.0) - float(offsets["approach_r_offset_mm"]),
        )
        self.assertEqual(in_front.y_mm, 140.0)
        self.assertEqual(lifted, TargetPose(230.0, 130.0, 60.0))
        self.assertAlmostEqual(
            hypot(retracted.x_mm, retracted.z_mm),
            hypot(230.0, 60.0) - float(offsets["approach_r_offset_mm"]),
        )
        self.assertEqual(retracted.y_mm, 130.0)


if __name__ == "__main__":
    unittest.main()
