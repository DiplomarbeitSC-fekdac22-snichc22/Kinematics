from __future__ import annotations

from math import pi
from pathlib import Path

import pytest

from api import RobotController
from config.config_loader import load_config
from simulator.coordinate_frames import robot_to_webots, webots_to_robot
from simulator.webots_motion_sink import WebotsMotionSink
from state_machine.pick_and_place import MotionCommand


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class FakePositionSensor:
    def __init__(self) -> None:
        self.value = 0.0
        self.period_ms: int | None = None

    def enable(self, period_ms: int) -> None:
        self.period_ms = period_ms

    def getValue(self) -> float:
        return self.value


class FakeMotor:
    def __init__(self, sensor: FakePositionSensor) -> None:
        self.sensor = sensor
        self.position = 0.0
        self.velocity = 0.0

    def setPosition(self, position: float) -> None:
        self.position = position

    def setVelocity(self, velocity: float) -> None:
        self.velocity = velocity


class FakeCamera:
    def __init__(self) -> None:
        self.period_ms: int | None = None

    def enable(self, period_ms: int) -> None:
        self.period_ms = period_ms

    def getWidth(self) -> int:
        return 640

    def getHeight(self) -> int:
        return 360


class FakeRangeFinder(FakeCamera):
    def getRangeImage(self) -> list[float]:
        return [0.5]


class FakeRobot:
    def __init__(self) -> None:
        self.devices: dict[str, object] = {}
        self.motors: list[FakeMotor] = []
        self.steps = 0

        for joint in (
            "J1_base",
            "J2_shoulder",
            "J3_elbow",
            "J4_wrist",
            "J5_gripper_left",
            "J5_gripper_right",
        ):
            sensor = FakePositionSensor()
            motor = FakeMotor(sensor)
            self.devices[joint] = motor
            self.devices[f"{joint}_sensor"] = sensor
            self.motors.append(motor)

        self.devices["camera_left"] = FakeCamera()
        self.devices["camera_right"] = FakeCamera()
        self.devices["tof_vl53l4cd"] = FakeRangeFinder()

    def getBasicTimeStep(self) -> float:
        return 16.0

    def getDevice(self, name: str) -> object | None:
        return self.devices.get(name)

    def step(self, time_step_ms: int) -> int:
        assert time_step_ms == 16
        self.steps += 1
        for motor in self.motors:
            motor.sensor.value = motor.position
        return 0


class ContactFakeRobot(FakeRobot):
    """Stop both closing jaws at the surface of the 50 mm demo ball."""

    def __init__(self) -> None:
        super().__init__()
        for joint in ("J5_gripper_left", "J5_gripper_right"):
            motor = self.devices[joint]
            sensor = self.devices[f"{joint}_sensor"]
            motor.position = 0.040
            sensor.value = 0.040

    def step(self, time_step_ms: int) -> int:
        assert time_step_ms == 16
        self.steps += 1

        for joint in (
            "J1_base",
            "J2_shoulder",
            "J3_elbow",
            "J4_wrist",
        ):
            motor = self.devices[joint]
            motor.sensor.value = motor.position

        for joint in ("J5_gripper_left", "J5_gripper_right"):
            motor = self.devices[joint]
            motor.sensor.value = max(0.030, motor.position)

        return 0


def test_coordinate_frame_round_trip() -> None:
    webots = robot_to_webots(230.0, 180.0, 60.0)
    assert webots == pytest.approx((0.230, -0.060, 0.320))
    assert webots_to_robot(*webots) == pytest.approx((230.0, 180.0, 60.0))


def test_webots_joint_and_gripper_mapping() -> None:
    robot = FakeRobot()
    sink = WebotsMotionSink(robot, config_dir=CONFIG_DIR)

    sink.send(
        MotionCommand(
            name="mapping_test",
            pulses_us={"J5_gripper": 1200},
            joint_angles_deg={
                "J1_base": 30.0,
                "J2_shoulder": 90.0,
                "J3_elbow": 90.0,
                "J4_wrist": -20.0,
            },
        )
    )

    assert robot.devices["J1_base"].position == pytest.approx(-pi / 6)
    assert robot.devices["J2_shoulder"].position == pytest.approx(pi / 2)
    assert robot.devices["J3_elbow"].position == pytest.approx(pi / 2)
    assert robot.devices["J4_wrist"].position == pytest.approx(-pi / 9)
    assert robot.devices["J5_gripper_left"].position == pytest.approx(0.040)
    assert robot.devices["J5_gripper_right"].position == pytest.approx(0.040)
    assert sink.sensor_snapshot()["tof_center_m"] == pytest.approx(0.5)


def test_webots_can_invert_recorded_pwm_when_angles_are_absent() -> None:
    robot = FakeRobot()
    sink = WebotsMotionSink(robot, config_dir=CONFIG_DIR)

    sink.send(
        MotionCommand(
            name="recorded_pwm",
            pulses_us={"J1_base": 1500, "J5_gripper": 1800},
        )
    )

    assert robot.devices["J1_base"].position == pytest.approx(0.0)
    assert robot.devices["J5_gripper_left"].position == pytest.approx(0.010)
    assert robot.devices["J5_gripper_right"].position == pytest.approx(0.010)


def test_gripper_accepts_stable_object_contact_before_fully_closed() -> None:
    robot = ContactFakeRobot()
    sink = WebotsMotionSink(robot, config_dir=CONFIG_DIR)

    sink.send(
        MotionCommand(
            name="close_gripper",
            pulses_us={"J5_gripper": 1800},
        )
    )

    assert robot.devices["J5_gripper_left_sensor"].value == pytest.approx(0.030)
    assert robot.devices["J5_gripper_right_sensor"].value == pytest.approx(0.030)


def test_repository_state_machine_completes_through_webots_sink() -> None:
    robot = FakeRobot()
    sink = WebotsMotionSink(robot, config_dir=CONFIG_DIR)
    controller = RobotController(sink)

    assert controller.run_pick_and_place(230.0, 180.0, 60.0)
    home = load_config("poses.toml", CONFIG_DIR)["poses"]["home"]
    assert robot.devices["J1_base"].position == pytest.approx(0.0)
    assert robot.devices["J2_shoulder"].position == pytest.approx(
        float(home["J2_shoulder"]) * pi / 180.0
    )
    assert robot.devices["J3_elbow"].position == pytest.approx(
        (180.0 - float(home["J3_elbow"])) * pi / 180.0
    )
    assert robot.devices["J4_wrist"].position == pytest.approx(
        float(home["J4_wrist"]) * pi / 180.0
    )


def test_webots_model_config_matches_primary_robot_configs() -> None:
    simulation = load_config("webots_simulation.toml", CONFIG_DIR)
    geometry = load_config("robot_geometry.toml", CONFIG_DIR)
    settings = load_config("kinematics_settings.toml", CONFIG_DIR)

    model = simulation["model"]
    links = geometry["link_lengths_mm"]

    assert model["link_1_mm"] == links["L1_shoulder_to_elbow"]
    assert model["link_2_mm"] == links["L2_elbow_to_wrist"]
    assert model["tool_length_mm"] == links["Lg_selected"]
    assert (
        model["maximum_gripper_opening_mm"]
        == geometry["gripper_geometry"]["max_opening_width_mm"]
    )
    assert (
        simulation["coordinate_mapping"]["top_reference_height_mm"]
        == settings["input_coordinates"]["max_height_mm"]
    )

    gripper = simulation["gripper"]
    opening_m = 2.0 * (
        gripper["open_slider_position_m"]
        - gripper["jaw_half_width_m"]
    )
    assert opening_m * 1000.0 == pytest.approx(
        model["maximum_gripper_opening_mm"]
    )
