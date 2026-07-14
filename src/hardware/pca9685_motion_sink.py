from pathlib import Path
from typing import Any

from config.config_loader import DEFAULT_CONFIG_DIR, load_config

_EXPECTED_JOINTS = (
    "J1_base",
    "J2_shoulder",
    "J3_elbow",
    "J4_wrist",
    "J5_gripper",
)

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

    def _validate_pwm_config(self):
        pass

    def _build_joint_outputs(self, pca_config, servo_config):
        pass

    def _create_default_hardware(self, pca_config):
        pass
