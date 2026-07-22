import csv
import json
import tomllib
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from calibration.models import JointCalibrationFit, PulseAngleMeasurement


_REQUIRED_COLUMNS = frozenset({"joint_name", "angle_deg", "pulse_us"})


def _parse_number(raw: str | None, *, field: str, row_number: int) -> float:
    if raw is None or not raw.strip():
        raise ValueError(f"CSV row {row_number}: {field} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number}: {field} must be numeric"
        ) from exc
    if not isfinite(value):
        raise ValueError(f"CSV row {row_number}: {field} must be finite")
    return value


def load_measurements_csv(path: Path | str) -> list[PulseAngleMeasurement]:
    """Load multiple measured pulse/angle pairs per joint from CSV."""
    path = Path(path)
    measurements: list[PulseAngleMeasurement] = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = _REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {tuple(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            joint_name = (row.get("joint_name") or "").strip()
            if not joint_name:
                raise ValueError(
                    f"CSV row {row_number}: joint_name is required"
                )
            measurements.append(
                PulseAngleMeasurement(
                    joint_name=joint_name,
                    angle_deg=_parse_number(
                        row.get("angle_deg"),
                        field="angle_deg",
                        row_number=row_number,
                    ),
                    pulse_us=_parse_number(
                        row.get("pulse_us"),
                        field="pulse_us",
                        row_number=row_number,
                    ),
                )
            )

    if not measurements:
        raise ValueError("CSV contains no measurement rows")

    return measurements


def load_servo_calibration(path: Path | str) -> dict[str, Any]:
    """Load the existing calibration solely as context for fitting."""
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def write_calibration_report(
    fits: Mapping[str, JointCalibrationFit],
    path: Path | str,
) -> None:
    """Write detailed fit results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0",
        "method": "ordinary_least_squares_pulse_from_angle",
        "calibration_status": "proposal_requires_manual_review",
        "joints": {
            joint_name: fit.as_dict()
            for joint_name, fit in sorted(fits.items())
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def _toml_number(value: float) -> str:
    return format(value, ".12g")


def write_calibration_proposal(
    fits: Mapping[str, JointCalibrationFit],
    path: Path | str,
) -> None:
    """Write reviewable TOML snippets without altering the live config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated from physical pulse/angle measurements.",
        "# This is a review proposal, not a replacement servo configuration.",
        "# Verify geometry, mechanical limits, pulse limits, and residuals first.",
        'calibration_status = "proposal_requires_manual_review"',
        "",
    ]

    for joint_name, fit in sorted(fits.items()):
        quoted_name = json.dumps(joint_name)
        lines.extend(
            (
                f"[joints.{quoted_name}]",
                f"theta_zero_deg = {_toml_number(fit.theta_zero_deg)}",
                f"direction = {fit.direction}",
                f"pulse_center_us = {_toml_number(fit.pulse_center_us)}",
                f"us_per_degree = {_toml_number(fit.us_per_degree)}",
                "",
                f"[diagnostics.{quoted_name}]",
                f"sample_count = {fit.sample_count}",
                f"unique_angle_count = {fit.unique_angle_count}",
                f"rmse_us = {_toml_number(fit.rmse_us)}",
                "mean_absolute_error_us = "
                f"{_toml_number(fit.mean_absolute_error_us)}",
                "maximum_absolute_error_us = "
                f"{_toml_number(fit.maximum_absolute_error_us)}",
                f"r_squared = {_toml_number(fit.r_squared)}",
                "observed_angle_min_deg = "
                f"{_toml_number(fit.observed_angle_min_deg)}",
                "observed_angle_max_deg = "
                f"{_toml_number(fit.observed_angle_max_deg)}",
                "observed_pulse_min_us = "
                f"{_toml_number(fit.observed_pulse_min_us)}",
                "observed_pulse_max_us = "
                f"{_toml_number(fit.observed_pulse_max_us)}",
                "",
            )
        )

    path.write_text("\n".join(lines), encoding="utf-8")
