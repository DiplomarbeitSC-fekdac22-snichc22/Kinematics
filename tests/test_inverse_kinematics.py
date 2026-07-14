import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinematics.inverse_kinematics import calculate_angles


class InverseKinematicsRegressionTests(unittest.TestCase):
    def test_current_pick_target_is_reachable(self) -> None:
        result = calculate_angles(230.0, 180.0, 60.0)

        self.assertTrue(result["reachable"])
        self.assertEqual(result["reasons"], [])
        self.assertAlmostEqual(result["angles_deg"]["base"], 14.6209, places=4)
        self.assertAlmostEqual(result["angles_deg"]["shoulder"], 101.9297, places=4)
        self.assertAlmostEqual(result["angles_deg"]["elbow"], 98.0622, places=4)
        self.assertAlmostEqual(result["angles_deg"]["wrist"], -19.9919, places=4)
        self.assertAlmostEqual(
            result["angles_deg"]["shoulder"]
            + result["angles_deg"]["elbow"]
            - 180.0
            + result["angles_deg"]["wrist"],
            0.0,
            places=7,
        )

    def test_target_beyond_front_operating_envelope_is_rejected(self) -> None:
        result = calculate_angles(240.1, 180.0, 60.0)

        self.assertFalse(result["reachable"])
        self.assertIn("x=240.1 mm outside workspace bounds", result["reasons"])


if __name__ == "__main__":
    unittest.main()
