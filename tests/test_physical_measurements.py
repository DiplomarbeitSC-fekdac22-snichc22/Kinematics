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
                enclosure["depth"] / 2 + clearance["front_extra"],
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
            {"front_extra": 20.0, "rear_extra": 20.0},
        )
        self.assertEqual(
            settings["shelving_mm"],
            {
                "first_shelf_from_top": 125.0,
                "compartment_count": 2,
                "compartment_height": 120.0,
                "floor_thickness": 5.0,
                "depth": 100.0,
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

    def test_servo_sheet_pulses_and_schematic_channels_are_recorded(self) -> None:
        joints = self.config["servo_calibration"]["joints"]

        for name in ("J1_base", "J2_shoulder", "J3_elbow"):
            joint = joints[name]
            self.assertEqual(
                (
                    joint["pulse_min_us"],
                    joint["pulse_center_us"],
                    joint["pulse_max_us"],
                ),
                (500, 1500, 2500),
            )
            self.assertAlmostEqual(
                joint["us_per_degree"],
                2000.0 / 270.0,
            )

        wrist = joints["J4_wrist"]
        self.assertEqual(wrist["pca9685_channel"], 8)
        self.assertEqual(
            (
                wrist["pulse_min_us"],
                wrist["pulse_center_us"],
                wrist["pulse_max_us"],
            ),
            (500, 1500, 2500),
        )
        self.assertAlmostEqual(
            wrist["us_per_degree"],
            2000.0 / 120.0,
        )

        gripper = joints["J5_gripper"]
        self.assertEqual(gripper["pca9685_channel"], 6)
        self.assertEqual(gripper["servo_type"], "MG995")
        self.assertEqual(
            (gripper["pulse_min_us"], gripper["pulse_max_us"]),
            (1000, 2000),
        )

    def test_vl53l4cd_webots_model_matches_product_sheet(self) -> None:
        simulation = self.config["webots_simulation"]

        self.assertEqual(
            simulation["devices"]["tof_range_finder"],
            "tof_vl53l4cd",
        )
        self.assertEqual(
            (
                simulation["tof"]["grid_width"],
                simulation["tof"]["grid_height"],
            ),
            (1, 1),
        )
        self.assertEqual(
            simulation["tof"]["diagonal_field_of_view_deg"],
            18.0,
        )
        self.assertEqual(simulation["tof"]["minimum_range_m"], 0.001)
        self.assertEqual(simulation["tof"]["maximum_range_m"], 1.2)


if __name__ == "__main__":
    unittest.main()
