from __future__ import annotations

from pathlib import Path

import pytest

from config.servo_compatibility import (
    ServoConfigurationError,
    compatibility_report,
    load_servo_configuration,
    main,
)


KINEMATICS_CONFIG = """
[defaults]
pwm_frequency_hz = 50

[joints.J1_base]
pca9685_channel = 0
pulse_min_us = 500
pulse_center_us = 1500
pulse_max_us = 2500
direction = 1
theta_min_deg = -90
theta_max_deg = 90
servo_type = "Miuzei_20kg_digital_270_degree"

[joints.J5_gripper]
function = "gripper"
pca9685_channel = 6
pulse_min_us = 1000
pulse_center_us = 1500
pulse_max_us = 2000
direction = 1
theta_min_deg = 0
theta_max_deg = 1
servo_type = "MG995"
pulse_open_us = 1200
pulse_closed_us = 1800
"""

CONTROLLER_CONFIG = """
[defaults]
pwm_frequency_hz = 50

[joints.J1_base]
channel = 0
pulse_min_us = 500
neutral_pulse_us = 1500
pulse_max_us = 2500
direction = 1
angle_min_deg = -90
angle_max_deg = 90
servo_model = "Miuzei_20kg_digital_270_degree"

[joints.J5_gripper]
function = "gripper"
channel = 6
pulse_min_us = 1000
neutral_pulse_us = 1500
pulse_max_us = 2000
direction = 1
angle_min_deg = 0
angle_max_deg = 1
servo_model = "MG995"
gripper_open_us = 1200
gripper_closed_us = 1800
"""


def _write_config(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_supported_schema_aliases_are_compatible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = load_servo_configuration(
        _write_config(tmp_path, "kinematics.toml", KINEMATICS_CONFIG)
    )
    right = load_servo_configuration(
        _write_config(tmp_path, "controller.toml", CONTROLLER_CONFIG)
    )

    assert compatibility_report(left, right) == 0
    assert "Result: COMPATIBLE" in capsys.readouterr().out


def test_cli_returns_one_for_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = _write_config(tmp_path, "kinematics.toml", KINEMATICS_CONFIG)
    right = _write_config(
        tmp_path,
        "controller.toml",
        CONTROLLER_CONFIG.replace("channel = 6", "channel = 8"),
    )

    exit_code = main([str(left), str(right)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "J5_gripper" in output
    assert "channel:           MISMATCH" in output
    assert "Kinematics:       6" in output
    assert "Robot-Controller: 8" in output
    assert "Result: INCOMPATIBLE" in output


def test_missing_joint_is_reported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = load_servo_configuration(
        _write_config(tmp_path, "kinematics.toml", KINEMATICS_CONFIG)
    )
    right = load_servo_configuration(
        _write_config(
            tmp_path,
            "controller.toml",
            CONTROLLER_CONFIG.split("[joints.J5_gripper]", maxsplit=1)[0],
        )
    )

    assert compatibility_report(left, right) == 1
    assert "MISSING FROM Robot-Controller" in capsys.readouterr().out


def test_duplicate_pca9685_channel_is_rejected(tmp_path: Path) -> None:
    invalid = _write_config(
        tmp_path,
        "controller.toml",
        CONTROLLER_CONFIG.replace("channel = 6", "channel = 0"),
    )

    with pytest.raises(ServoConfigurationError, match="reuses PCA9685 channel 0"):
        load_servo_configuration(invalid)


def test_cli_returns_two_for_invalid_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = _write_config(tmp_path, "kinematics.toml", KINEMATICS_CONFIG)
    invalid = _write_config(tmp_path, "invalid.toml", "not valid toml =")

    exit_code = main([str(left), str(invalid)])

    assert exit_code == 2
    assert "ERROR:" in capsys.readouterr().err
