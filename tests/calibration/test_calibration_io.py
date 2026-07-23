import json
import tomllib
from pathlib import Path

import pytest

from calibration.cli import main
from calibration.io import load_measurements_csv


SERVO_CONFIG = """
[joints.J1_base]
theta_zero_deg = 0.0
"""


MEASUREMENTS = """joint_name,angle_deg,pulse_us
J1_base,-30,1200
J1_base,0,1500
J1_base,30,1800
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_multiple_pairs_per_joint(tmp_path: Path) -> None:
    measurements = load_measurements_csv(
        _write(tmp_path / "measurements.csv", MEASUREMENTS)
    )

    assert len(measurements) == 3
    assert {measurement.joint_name for measurement in measurements} == {
        "J1_base"
    }
    assert [measurement.angle_deg for measurement in measurements] == [
        -30.0,
        0.0,
        30.0,
    ]


def test_rejects_incomplete_measurement_row(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "measurements.csv",
        "joint_name,angle_deg,pulse_us\nJ1_base,0,\n",
    )

    with pytest.raises(ValueError, match="pulse_us is required"):
        load_measurements_csv(path)


def test_cli_writes_review_report_and_toml_proposal(
    tmp_path: Path,
) -> None:
    measurements = _write(tmp_path / "measurements.csv", MEASUREMENTS)
    config = _write(tmp_path / "servo_calibration.toml", SERVO_CONFIG)
    report_path = tmp_path / "fit.json"
    proposal_path = tmp_path / "proposal.toml"

    exit_code = main(
        [
            str(measurements),
            "--config",
            str(config),
            "--report-json",
            str(report_path),
            "--proposal-toml",
            str(proposal_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["calibration_status"] == "proposal_requires_manual_review"
    suggested = report["joints"]["J1_base"]["suggested_updates"]
    assert suggested["direction"] == 1
    assert suggested["pulse_center_us"] == pytest.approx(1500.0)
    assert suggested["us_per_degree"] == pytest.approx(10.0)

    proposal = tomllib.loads(proposal_path.read_text(encoding="utf-8"))
    assert proposal["calibration_status"] == "proposal_requires_manual_review"
    assert proposal["joints"]["J1_base"]["direction"] == 1
    assert proposal["diagnostics"]["J1_base"]["sample_count"] == 3
    assert "requires_physical_calibration = false" not in proposal_path.read_text(
        encoding="utf-8"
    )


def test_cli_does_not_write_outputs_for_invalid_data(tmp_path: Path) -> None:
    measurements = _write(
        tmp_path / "measurements.csv",
        "joint_name,angle_deg,pulse_us\nJ1_base,0,1500\n",
    )
    config = _write(tmp_path / "servo_calibration.toml", SERVO_CONFIG)
    report_path = tmp_path / "fit.json"
    proposal_path = tmp_path / "proposal.toml"

    exit_code = main(
        [
            str(measurements),
            "--config",
            str(config),
            "--report-json",
            str(report_path),
            "--proposal-toml",
            str(proposal_path),
        ]
    )

    assert exit_code == 2
    assert not report_path.exists()
    assert not proposal_path.exists()
