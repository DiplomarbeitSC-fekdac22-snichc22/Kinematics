from dataclasses import dataclass
from math import isclose
from numbers import Integral
from pathlib import Path
from typing import Any

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from state_machine.pick_and_place import MotionCommand

_EXPECTED_JOINTS = (
    "J1_base",
    "J2_shoulder",
    "J3_elbow",
    "J4_wrist",
    "J5_gripper",
)

_PCA9685_CHANNEL_COUNT = 16
_PCA9685_RESOLUTION_COUNT = 4096
_CIRCUITPYTHON_DUTY_CYCLE_STEPS = 65536


@dataclass(frozen=True)
class _JointOutputs:
    channel: int
    safe_min_us: float
    safe_max_us: float


class Pca9685MotionSink:
    """Send servo pulse commands to the PCA9685 board."""

    def __init__(
            self,
            pca: Any | None = None,
            *,
            config_dir: Path | str = DEFAULT_CONFIG_DIR
    ) -> None:
        pca_config = load_config("pca9685.toml", config_dir)
        servo_config = load_config("servo_calibration.toml", config_dir)

        self._frequency_hz = float(pca_config["frequency_hz"])
        self._resolution_counts = int(pca_config["resolution_counts"])
        self._period_us = float(pca_config["period_us"])
        self._channel_count = int(pca_config["channel_count"])

        self._validate_pwm_config()

        self._joint_outputs = self._build_joint_outputs(
            pca_config=pca_config,
            servo_config=servo_config,
        )

        self._i2c: Any | None = None
        self._owns_hardware = pca is None
        self._closed = False

        if pca is None:
            self._i2c, self._pca = self._create_default_hardware(pca_config)
        else:
            self._pca = pca

        self._pca.frequency = self._frequency_hz

    @property
    def channel_map(self) -> dict[str, int]:
        """Return a copy of the configured joint to PCA9685 channel mapping."""
        return {
            joint_name: output.channel
            for joint_name, output in self._joint_outputs.items()
        }

    @property
    def frequency_hz(self) -> float:
        """Return the configured PCA9685 PWM frequency."""
        return self._frequency_hz

    def send(self, command: MotionCommand) -> None:
        """Validate and send one command."""
        self._ensure_open()

        writes: list[tuple[int, int]] = []

        for joint_name, pulse_us in command.pulses_us.items():
            output = self._joint_outputs.get(joint_name)

            if output is None:
                raise KeyError(f"Unknown servo output: {joint_name}")

            if isinstance(pulse_us, bool) or not isinstance(pulse_us, Integral):
                raise TypeError(
                    f"Pulse for {joint_name} must be an integer of "
                    f"microseconds; got {type(pulse_us).__name__}"
                )

            duty_cycle = self.pulse_us_to_duty_cycle(pulse_us)

            writes.append((output.channel, duty_cycle))

        for channel, duty_cycle in writes:
            self._pca.channels[channel].duty_cycle = duty_cycle

    def pulse_us_to_duty_cycle(self, pulse_us: int) -> int:
        """Convert a pulse width in microseconds to a PCA9685 duty cycle."""
        pulse_count = round(pulse_us / self._period_us * self._resolution_counts)

        if not 0 <= pulse_count < self._resolution_counts:
            raise ValueError(f"Pulse width produces invalid count: {pulse_us} us -> {pulse_count}")

        scale = (_CIRCUITPYTHON_DUTY_CYCLE_STEPS // self._resolution_counts)

        return pulse_count * scale

    def disable_all(self) -> None:
        """Disable all servo outputs."""
        self._ensure_open()

        for channel in range(self._channel_count):
            self._pca.channels[channel].duty_cycle = 0

    def close(self) -> None:
        """Disable outputs and release hardware resources."""
        if self._closed:
            return

        try:
            self.disable_all()
        finally:
            try:
                if self._owns_hardware:
                    self._pca.deinit()
            finally:
                if self._owns_hardware and self._i2c is not None:
                    self._i2c.deinit()

                self._closed = True

    def __enter__(self) -> "Pca9685MotionSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_pwm_config(self):
        if self._frequency_hz <= 0:
            raise ValueError("PCA9685 frequency must be positive")

        if self._resolution_counts != _PCA9685_RESOLUTION_COUNT:
            raise ValueError(f"PCA9685 resolution count must be {_PCA9685_RESOLUTION_COUNT}")

        if not 1 <= self._channel_count <= _PCA9685_CHANNEL_COUNT:
            raise ValueError(f"PCA9685 channel count must be between 1 and {_PCA9685_CHANNEL_COUNT}")

        if _CIRCUITPYTHON_DUTY_CYCLE_STEPS % self._resolution_counts != 0:
            raise ValueError("PCA9685 resolution count must evenly divide the CircuitPython duty cycle steps")

    def _build_joint_outputs(
            self,
            pca_config: dict[str, Any],
            servo_config: dict[str, Any]
    ) -> dict[str, _JointOutputs]:
        channel_map = pca_config["channel_map"]
        joints = servo_config["joints"]
        defaults = servo_config["defaults"]

        missing_channels = set(_EXPECTED_JOINTS) - set(channel_map)
        missing_calibration = set(_EXPECTED_JOINTS) - set(joints)

        if missing_channels:
            raise KeyError(f"Missing PCA9685 channel mappings: {missing_channels}")

        if missing_calibration:
            raise KeyError(f"Missing PCA9685 calibration entries: {missing_calibration}")

        servo_frequency_hz = float(defaults["pwm_frequency_hz"])

        if not isclose(
                servo_frequency_hz,
                self._frequency_hz,
                rel_tol=0.0,
                abs_tol=0.01,
        ):
            raise ValueError(
                "PWM frequency mismatch: pca9685.toml uses "
                f"{self._frequency_hz} Hz; "
                "servo_calibration.toml uses "
                f"{servo_frequency_hz} Hz"
            )

        electrical_min = int(defaults["pulse_electrical_min_us"])
        electrical_max = int(defaults["pulse_electrical_max_us"])

        use_initial_range = bool(defaults.get("clamp_to_initial_safe_range", False))

        initial_min = int(defaults["pulse_initial_safe_min_us"])
        initial_max = int(defaults["pulse_initial_safe_max_us"])

        outputs: dict[str, _JointOutputs] = {}
        used_channels: set[int] = set()

        for joint_name in _EXPECTED_JOINTS:
            joint_config = joints[joint_name]

            channel = int(channel_map[joint_name])
            calibration_channel = int(joint_config["pca9685_channel"])

            if channel != calibration_channel:
                raise ValueError(
                    f"Channel mismatch for {joint_name}: "
                    f"pca9685.toml uses {channel}; "
                    f"servo_calibration.toml uses {calibration_channel}"
                )

            if not 0 <= channel < self._channel_count:
                raise ValueError(f"Configured channel for {joint_name} is out of range")

            if channel in used_channels:
                raise ValueError(f"PCA9685 channel {channel} is already used")

            used_channels.add(channel)

            safe_min = max(
                int(joint_config["pulse_min_us"]),
                electrical_min,
            )

            safe_max = min(
                int(joint_config["pulse_max_us"]),
                electrical_max,
            )

            if use_initial_range:
                safe_min = max(safe_min, initial_min)
                safe_max = min(safe_max, initial_max)

            if safe_min > safe_max:
                raise ValueError(f"Configured pulse ranges do not overlap for {joint_name}")

            outputs[joint_name] = _JointOutputs(
                channel=channel,
                safe_min_us=safe_min,
                safe_max_us=safe_max
            )

        return outputs

    @staticmethod
    def _create_default_hardware(pca_config: dict[str, Any]) -> tuple[Any, Any]:
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
        except ImportError as exc:
            raise RuntimeError(
                "PCA9685 hardware support is not installed. "
                "Install the project with the 'hardware' extra."
            ) from exc

        i2c = busio.I2C(board.SCL, board.SDA)

        try:
            pca = PCA9685(
                i2c,
                address=int(pca_config.get("i2c_address", 0x40)),
            )
        except Exception:
            i2c.deinit()
            raise

        return i2c, pca

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Pca9685MotionSink is closed")
