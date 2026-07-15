"""Webots implementation of the state machine's motion-command sink."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, radians
from pathlib import Path
from typing import Any, Protocol

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from state_machine.pick_and_place import MotionCommand


class WebotsSimulationEnded(RuntimeError):
    """Raised when Webots stops while a command is still executing."""


class _Motor(Protocol):
    def setPosition(self, position: float) -> None: ...

    def setVelocity(self, velocity: float) -> None: ...


class _PositionSensor(Protocol):
    def enable(self, period_ms: int) -> None: ...

    def getValue(self) -> float: ...


class _Sensor(Protocol):
    def enable(self, period_ms: int) -> None: ...


class _Robot(Protocol):
    def getBasicTimeStep(self) -> float: ...

    def getDevice(self, name: str) -> Any: ...

    def step(self, time_step_ms: int) -> int: ...


@dataclass(frozen=True)
class _JointBinding:
    motor: _Motor
    sensor: _PositionSensor
    sign: float
    offset_deg: float
    tolerance: float


@dataclass(frozen=True)
class _SettleTarget:
    name: str
    sensor: _PositionSensor
    position: float
    tolerance: float
    initial_position: float
    allow_contact: bool = False
    contact_minimum_motion: float = 0.0
    contact_stability_delta: float = 0.0


@dataclass(frozen=True)
class _GripperBinding:
    motors: tuple[_Motor, _Motor]
    sensors: tuple[_PositionSensor, _PositionSensor]
    open_position: float
    closed_position: float
    tolerance: float
    open_pulse_us: int
    closed_pulse_us: int
    contact_minimum_motion: float
    contact_stability_delta: float


class WebotsMotionSink:
    """Drive Webots motors from the same commands used by the PCA9685 sink.

    Arm joints use the exact mathematical angles attached to ``MotionCommand``.
    PWM inversion remains as a fallback for recorded or externally constructed
    commands. The gripper intentionally uses its calibrated open/closed pulses,
    because J5 is a command rather than an inverse-kinematics angle.
    """

    def __init__(
        self,
        robot: _Robot,
        *,
        config_dir: Path | str = DEFAULT_CONFIG_DIR,
    ) -> None:
        self.robot = robot
        self.config_dir = Path(config_dir)
        self.simulation = load_config(
            "webots_simulation.toml",
            self.config_dir,
        )
        self.servo = load_config(
            "servo_calibration.toml",
            self.config_dir,
        )
        self.poses = load_config(
            "poses.toml",
            self.config_dir,
        )

        configured_step = int(
            self.simulation["timing"]["basic_time_step_ms"]
        )
        reported_step = int(round(float(robot.getBasicTimeStep())))
        self.time_step_ms = reported_step or configured_step

        if self.time_step_ms != configured_step:
            raise ValueError(
                "Webots basicTimeStep does not match "
                f"webots_simulation.toml: {self.time_step_ms} != "
                f"{configured_step} ms"
            )

        self.command_timeout_s = float(
            self.simulation["timing"]["command_timeout_s"]
        )
        self.settle_time_s = float(
            self.simulation["timing"]["settle_time_s"]
        )

        self.joints = self._bind_arm_joints()
        self.gripper = self._bind_gripper()
        self.cameras, self.tof = self._enable_environment_sensors()

        # Advance once so enabled PositionSensors have their first samples.
        self._step_or_raise()

    def send(self, command: MotionCommand) -> None:
        """Set all targets atomically and wait until they settle."""
        targets: list[_SettleTarget] = []

        for joint_name, binding in self.joints.items():
            target_angle = self._command_angle(command, joint_name)

            if target_angle is None:
                continue

            webots_position = radians(
                binding.offset_deg + binding.sign * target_angle
            )
            targets.append(
                _SettleTarget(
                    name=joint_name,
                    sensor=binding.sensor,
                    position=webots_position,
                    tolerance=binding.tolerance,
                    initial_position=float(binding.sensor.getValue()),
                )
            )
            binding.motor.setPosition(webots_position)

        gripper_joint = str(
            self.simulation["gripper"]["source_joint"]
        )
        if gripper_joint in command.pulses_us:
            pulse_us = int(command.pulses_us[gripper_joint])
            position = self._gripper_position(pulse_us)
            allow_contact = pulse_us == self.gripper.closed_pulse_us

            for index, (motor, sensor) in enumerate(
                zip(
                    self.gripper.motors,
                    self.gripper.sensors,
                    strict=True,
                )
            ):
                targets.append(
                    _SettleTarget(
                        name=(
                            "J5_gripper_left"
                            if index == 0
                            else "J5_gripper_right"
                        ),
                        sensor=sensor,
                        position=position,
                        tolerance=self.gripper.tolerance,
                        initial_position=float(sensor.getValue()),
                        allow_contact=allow_contact,
                        contact_minimum_motion=(
                            self.gripper.contact_minimum_motion
                        ),
                        contact_stability_delta=(
                            self.gripper.contact_stability_delta
                        ),
                    )
                )
                motor.setPosition(position)

        if targets:
            self._wait_until_settled(command.name, targets)

    def sensor_snapshot(self) -> dict[str, object]:
        """Return lightweight metadata plus the centre ToF range sample."""
        snapshot: dict[str, object] = {
            "cameras": [
                {
                    "name": name,
                    "width": int(device.getWidth()),
                    "height": int(device.getHeight()),
                }
                for name, device in self.cameras.items()
            ]
        }

        range_image = self.tof.getRangeImage()
        if range_image:
            centre = len(range_image) // 2
            value = float(range_image[centre])
            snapshot["tof_center_m"] = value if isfinite(value) else None
        else:
            snapshot["tof_center_m"] = None

        return snapshot

    def _bind_arm_joints(self) -> dict[str, _JointBinding]:
        bindings: dict[str, _JointBinding] = {}

        for joint_name, config in self.simulation["joints"].items():
            motor = self._require_device(str(config["motor_device"]))
            sensor = self._require_device(str(config["sensor_device"]))
            motor.setVelocity(float(config["max_velocity_rad_s"]))
            sensor.enable(self.time_step_ms)

            bindings[joint_name] = _JointBinding(
                motor=motor,
                sensor=sensor,
                sign=float(config["kinematic_to_webots_sign"]),
                offset_deg=float(
                    config["kinematic_to_webots_offset_deg"]
                ),
                tolerance=float(config["position_tolerance_rad"]),
            )

        return bindings

    def _bind_gripper(self) -> _GripperBinding:
        config = self.simulation["gripper"]
        commands = self.poses["gripper_commands"]

        motors = (
            self._require_device(str(config["left_motor_device"])),
            self._require_device(str(config["right_motor_device"])),
        )
        sensors = (
            self._require_device(str(config["left_sensor_device"])),
            self._require_device(str(config["right_sensor_device"])),
        )

        for motor in motors:
            motor.setVelocity(float(config["max_velocity_m_s"]))
        for sensor in sensors:
            sensor.enable(self.time_step_ms)

        return _GripperBinding(
            motors=motors,
            sensors=sensors,
            open_position=float(config["open_slider_position_m"]),
            closed_position=float(config["closed_slider_position_m"]),
            tolerance=float(config["position_tolerance_m"]),
            open_pulse_us=int(commands["open_pulse_us"]),
            closed_pulse_us=int(commands["closed_pulse_us"]),
            contact_minimum_motion=float(
                config["contact_minimum_motion_m"]
            ),
            contact_stability_delta=float(
                config["contact_stability_delta_m"]
            ),
        )

    def _enable_environment_sensors(
        self,
    ) -> tuple[dict[str, Any], Any]:
        config = self.simulation["devices"]
        period = int(config["sensor_period_ms"])
        cameras: dict[str, Any] = {}

        for key in ("left_camera", "right_camera"):
            name = str(config[key])
            device: _Sensor = self._require_device(name)
            device.enable(period)
            cameras[name] = device

        tof = self._require_device(str(config["tof_range_finder"]))
        tof.enable(period)

        return cameras, tof

    def _command_angle(
        self,
        command: MotionCommand,
        joint_name: str,
    ) -> float | None:
        if (
            command.joint_angles_deg is not None
            and joint_name in command.joint_angles_deg
        ):
            return float(command.joint_angles_deg[joint_name])

        if joint_name not in command.pulses_us:
            return None

        joint = self.servo["joints"][joint_name]
        direction = float(joint["direction"])
        scale = float(joint["us_per_degree"])

        if direction == 0.0 or scale == 0.0:
            raise ValueError(
                f"Cannot invert PWM calibration for {joint_name}"
            )

        return float(joint["theta_zero_deg"]) + (
            float(command.pulses_us[joint_name])
            - float(joint["pulse_center_us"])
        ) / (direction * scale)

    def _gripper_position(self, pulse_us: int) -> float:
        pulse_span = (
            self.gripper.closed_pulse_us
            - self.gripper.open_pulse_us
        )

        if pulse_span == 0:
            raise ValueError("Gripper open and closed pulses must differ")

        fraction_closed = (
            pulse_us - self.gripper.open_pulse_us
        ) / pulse_span
        fraction_closed = max(0.0, min(1.0, fraction_closed))

        return self.gripper.open_position + fraction_closed * (
            self.gripper.closed_position
            - self.gripper.open_position
        )

    def _wait_until_settled(
        self,
        command_name: str,
        targets: list[_SettleTarget],
    ) -> None:
        maximum_steps = max(
            1,
            ceil(
                self.command_timeout_s
                * 1000.0
                / self.time_step_ms
            ),
        )
        settled_steps_required = max(
            1,
            ceil(
                self.settle_time_s
                * 1000.0
                / self.time_step_ms
            ),
        )
        settled_steps = 0
        previous_positions = [
            target.initial_position
            for target in targets
        ]

        for _ in range(maximum_steps):
            self._step_or_raise()

            positions = [
                float(target.sensor.getValue())
                for target in targets
            ]

            if all(
                self._target_is_settled(
                    target,
                    position,
                    previous_position,
                )
                for target, position, previous_position in zip(
                    targets,
                    positions,
                    previous_positions,
                    strict=True,
                )
            ):
                settled_steps += 1
                if settled_steps >= settled_steps_required:
                    return
            else:
                settled_steps = 0

            previous_positions = positions

        unresolved = [
            (
                f"{target.name}: actual={position:.4f}, "
                f"target={target.position:.4f}, "
                f"error={abs(position - target.position):.4f}"
            )
            for target, position in zip(
                targets,
                previous_positions,
                strict=True,
            )
            if not self._target_is_settled(
                target,
                position,
                position,
            )
        ]
        raise TimeoutError(
            f"Webots command {command_name!r} did not settle within "
            f"{self.command_timeout_s:.1f} s; "
            + "; ".join(unresolved)
        )

    @staticmethod
    def _target_is_settled(
        target: _SettleTarget,
        position: float,
        previous_position: float,
    ) -> bool:
        if not isfinite(position):
            return False

        if abs(position - target.position) <= target.tolerance:
            return True

        if not target.allow_contact:
            return False

        requested_motion = target.position - target.initial_position
        if requested_motion == 0.0:
            return False

        direction = 1.0 if requested_motion > 0.0 else -1.0
        progress = (
            position - target.initial_position
        ) * direction
        stable_against_contact = (
            abs(position - previous_position)
            <= target.contact_stability_delta
        )

        return (
            progress >= target.contact_minimum_motion
            and stable_against_contact
        )

    def _step_or_raise(self) -> None:
        if self.robot.step(self.time_step_ms) == -1:
            raise WebotsSimulationEnded(
                "Webots stopped while executing a robot command"
            )

    def _require_device(self, name: str) -> Any:
        device = self.robot.getDevice(name)
        if device is None:
            raise KeyError(f"Webots device not found: {name}")
        return device
