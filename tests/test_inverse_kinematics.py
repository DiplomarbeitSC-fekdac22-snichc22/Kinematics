import sys
import unittest
from math import degrees
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinematics.inverse_kinematics import (
    calculate_angle_solutions,
    calculate_angles,
)
from kinematics.forward_kinematics import calculate_gripper_center
from kinematics.singularity_analyzer import analyze_configuration


class InverseKinematicsRegressionTests(unittest.TestCase):
    def test_returns_both_mathematical_branches(self) -> None:
        solutions = calculate_angle_solutions(230.0, 180.0, 60.0)

        self.assertEqual(
            [solution["branch"] for solution in solutions],
            ["elbow_back", "elbow_forward"],
        )
        self.assertEqual(
            {solution["elbow_relative_sign"] for solution in solutions},
            {-1.0, 1.0},
        )

        elbow_back, elbow_forward = solutions
        self.assertAlmostEqual(
            elbow_back["elbow_relative_angle_deg"],
            -elbow_forward["elbow_relative_angle_deg"],
            places=7,
        )
        self.assertAlmostEqual(
            elbow_back["angles_deg"]["base"],
            elbow_forward["angles_deg"]["base"],
            places=7,
        )
        self.assertAlmostEqual(
            elbow_back["angles_deg"]["elbow"],
            elbow_forward["angles_deg"]["elbow"],
            places=7,
        )
        self.assertNotAlmostEqual(
            elbow_back["angles_deg"]["shoulder"],
            elbow_forward["angles_deg"]["shoulder"],
            places=7,
        )

        for solution in solutions:
            angles = solution["angles_deg"]
            self.assertAlmostEqual(
                angles["shoulder"]
                + solution["elbow_relative_angle_deg"]
                + angles["wrist"],
                0.0,
                places=7,
            )

    def test_returns_branch_even_when_it_violates_joint_limits(self) -> None:
        elbow_back, elbow_forward = calculate_angle_solutions(
            230.0,
            180.0,
            60.0,
        )

        self.assertTrue(elbow_back["reachable"])
        self.assertFalse(elbow_forward["reachable"])
        self.assertAlmostEqual(
            elbow_forward["angles_deg"]["shoulder"],
            8.1582,
            places=4,
        )
        self.assertAlmostEqual(
            elbow_forward["angles_deg"]["wrist"],
            121.2998,
            places=4,
        )
        self.assertTrue(
            any(
                reason.startswith("J4_wrist angle")
                for reason in elbow_forward["reason_groups"]["joint_limits"]
            )
        )

    def test_singularity_analysis_uses_each_candidate_branch_sign(self) -> None:
        for solution in calculate_angle_solutions(230.0, 180.0, 60.0):
            angles = solution["angles_deg"]
            joint_angles = {
                "J1_base": angles["base"],
                "J2_shoulder": angles["shoulder"],
                "J3_elbow": angles["elbow"],
                "J4_wrist": angles["wrist"],
            }

            analysis = analyze_configuration(
                joint_angles,
                elbow_relative_sign=solution["elbow_relative_sign"],
            )

            self.assertAlmostEqual(
                degrees(analysis.jacobian.elbow_relative_angle_rad),
                solution["elbow_relative_angle_deg"],
                places=7,
            )

    def test_current_pick_target_is_reachable(self) -> None:
        result = calculate_angles(230.0, 180.0, 60.0)

        self.assertEqual(result["branch"], "elbow_back")
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
