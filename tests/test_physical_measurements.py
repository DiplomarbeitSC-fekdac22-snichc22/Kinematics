import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.config_loader import load_configs


class PhysicalMeasurementsConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_configs(ROOT / "configs")

    def test_measured_enclosure_matches_runtime_coordinates_and_bounds(self) -> None:
        settings = self.config["kinematics_settings"]
        enclosure = settings["physical_enclosure_mm"]
        clearance = settings["elbow_clearance_mm"]
        coordinates = settings["input_coordinates"]
        bounds = settings["workspace_bounds_robot_base_mm"]

        self.assertEqual(enclosure, {"height": 500.0, "depth": 400.0, "width": 360.0})
        self.assertEqual(coordinates["max_height_mm"], enclosure["height"])
        self.assertEqual(
            coordinates["base_rotation_axis_at_mounting_plate_mm"],
            [0.0, enclosure["height"], 0.0],
        )
        self.assertEqual(
            (bounds["x_min"], bounds["x_max"]),
            (
                -enclosure["depth"] / 2 - clearance["rear_extra"],
                enclosure["depth"] / 2 + clearance["front_extra_max"],
            ),
        )
        self.assertEqual((bounds["y_min"], bounds["y_max"]), (0.0, enclosure["height"]))
        self.assertEqual(
            (bounds["z_min"], bounds["z_max"]),
            (-enclosure["width"] / 2, enclosure["width"] / 2),
        )
        self.assertEqual(coordinates["x_positive_direction"], "forward")
        self.assertEqual(coordinates["z_positive_direction"], "right")

    def test_elbow_clearance_and_shelves_match_measurements(self) -> None:
        settings = self.config["kinematics_settings"]

        self.assertEqual(
            settings["elbow_clearance_mm"],
            {"front_extra_min": 20.0, "front_extra_max": 40.0, "rear_extra": 20.0},
        )
        self.assertEqual(
            settings["shelving_mm"],
            {
                "first_shelf_from_top": 125.0,
                "compartment_height": 120.0,
                "floor_thickness": 5.0,
                "x_position_mm": 100.0,
                "x_direction": "positive",
            },
        )

    def test_gripper_and_drop_off_match_measurements(self) -> None:
        geometry = self.config["robot_geometry"]
        drop_off = self.config["poses"]["cartesian_targets"]["drop_off"]

        self.assertEqual(geometry["gripper_geometry"]["max_opening_width_mm"], 70.0)
        self.assertEqual(drop_off["x_mm"], 0.0)
        self.assertEqual(drop_off["y_mm"], 370.0)
        self.assertEqual(drop_off["z_mm"], 130.0)
        self.assertEqual(drop_off["z_direction"], "right")
        self.assertNotIn("requires_x_measurement", drop_off)


if __name__ == "__main__":
    unittest.main()
