import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinematics.inverse_kinematics import calculate_angles
from kinematics.forward_kinematics import calculate_gripper_center


class InverseKinematicsRegressionTests(unittest.TestCase):
    def test_current_pick_target_is_reachable(self) -> None:
        result = calculate_angles(230.0, 180.0, 60.0)

        self.assertTrue(result["reachable"])
        self.assertEqual(result["reasons"], [])
        self.assertAlmostEqual(result["angles_deg"]["base"], 14.6209, places=4)
        self.assertAlmostEqual(result["angles_deg"]["shoulder"], -97.6350, places=4)
        self.assertAlmostEqual(result["angles_deg"]["elbow"], 50.5419, places=4)
        self.assertAlmostEqual(result["angles_deg"]["wrist"], -31.8231, places=4)
        self.assertAlmostEqual(
            result["angles_deg"]["shoulder"]
            + 180.0
            - result["angles_deg"]["elbow"]
            + result["angles_deg"]["wrist"],
            0.0,
            places=7,
        )

    def test_target_inside_open_shelf_compartment_is_reachable(self) -> None:
        result = calculate_angles(230.0, 180.0, 60.0)

        self.assertTrue(result["reachable"], result["reasons"])

    def test_target_in_shelf_floor_is_rejected(self) -> None:
        result = calculate_angles(230.0, 247.0, 60.0)

        self.assertFalse(result["reachable"])
        self.assertIn(
            "x=230.0 mm enters shelf depth but y=247.0 mm "
            "intersects a shelf floor or closed shelf region",
            result["reasons"],
        )

    def test_target_beyond_shelf_depth_is_rejected(self) -> None:
        result = calculate_angles(320.1, 180.0, 60.0)

        self.assertFalse(result["reachable"])
        self.assertIn("x=320.1 mm outside workspace bounds", result["reasons"])

    def test_inverse_and_forward_kinematics_round_trip(self) -> None:
        for target in (
                (230.0, 180.0, 60.0),
                (0.0, 370.0, 130.0),
        ):
            with self.subTest(target=target):
                result = calculate_angles(*target)
                self.assertTrue(result["reachable"], result["reasons"])
                angles = result["angles_deg"]

                reconstructed = calculate_gripper_center(
                    {
                        "J1_base": angles["base"],
                        "J2_shoulder": angles["shoulder"],
                        "J3_elbow": angles["elbow"],
                        "J4_wrist": angles["wrist"],
                    }
                )

                self.assertAlmostEqual(reconstructed["x_mm"], target[0], places=7)
                self.assertAlmostEqual(reconstructed["y_mm"], target[1], places=7)
                self.assertAlmostEqual(reconstructed["z_mm"], target[2], places=7)


if __name__ == "__main__":
    unittest.main()
