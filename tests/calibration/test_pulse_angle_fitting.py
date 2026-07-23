import pytest

from calibration.fitting import (
    CalibrationDataError,
    fit_joint_calibration,
    fit_servo_calibrations,
)
from calibration.models import PulseAngleMeasurement


def _measurement(
    joint_name: str,
    angle_deg: float,
    pulse_us: float,
) -> PulseAngleMeasurement:
    return PulseAngleMeasurement(joint_name, angle_deg, pulse_us)


def test_fits_positive_direction_from_multiple_pairs() -> None:
    fit = fit_joint_calibration(
        "J1_base",
        [
            _measurement("J1_base", -30.0, 1200.0),
            _measurement("J1_base", 0.0, 1500.0),
            _measurement("J1_base", 30.0, 1800.0),
            _measurement("J1_base", 60.0, 2100.0),
        ],
        theta_zero_deg=0.0,
    )

    assert fit.direction == 1
    assert fit.us_per_degree == pytest.approx(10.0)
    assert fit.pulse_center_us == pytest.approx(1500.0)
    assert fit.rmse_us == pytest.approx(0.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.sample_count == 4
    assert fit.unique_angle_count == 4


def test_fits_negative_direction_at_configured_zero() -> None:
    fit = fit_joint_calibration(
        "J3_elbow",
        [
            _measurement("J3_elbow", 60.0, 1800.0),
            _measurement("J3_elbow", 90.0, 1500.0),
            _measurement("J3_elbow", 120.0, 1200.0),
        ],
        theta_zero_deg=90.0,
    )

    assert fit.direction == -1
    assert fit.slope_us_per_degree == pytest.approx(-10.0)
    assert fit.us_per_degree == pytest.approx(10.0)
    assert fit.pulse_center_us == pytest.approx(1500.0)
    assert fit.predicted_pulse_us(105.0) == pytest.approx(1350.0)


def test_reports_residual_diagnostics_for_noisy_measurements() -> None:
    fit = fit_joint_calibration(
        "J4_wrist",
        [
            _measurement("J4_wrist", -30.0, 1198.0),
            _measurement("J4_wrist", 0.0, 1503.0),
            _measurement("J4_wrist", 30.0, 1799.0),
            _measurement("J4_wrist", 60.0, 2102.0),
        ],
        theta_zero_deg=0.0,
    )

    assert fit.rmse_us > 0.0
    assert fit.maximum_absolute_error_us >= fit.mean_absolute_error_us
    assert 0.99 < fit.r_squared < 1.0


def test_rejects_too_few_or_repeated_angles() -> None:
    with pytest.raises(CalibrationDataError, match="at least 3 measured pairs"):
        fit_joint_calibration(
            "J1_base",
            [
                _measurement("J1_base", 0.0, 1500.0),
                _measurement("J1_base", 30.0, 1800.0),
            ],
            theta_zero_deg=0.0,
        )

    with pytest.raises(CalibrationDataError, match="3 distinct measured angles"):
        fit_joint_calibration(
            "J1_base",
            [
                _measurement("J1_base", 0.0, 1499.0),
                _measurement("J1_base", 0.0, 1501.0),
                _measurement("J1_base", 30.0, 1800.0),
            ],
            theta_zero_deg=0.0,
        )


def test_rejects_unknown_joint() -> None:
    with pytest.raises(CalibrationDataError, match="unknown joints"):
        fit_servo_calibrations(
            [
                _measurement("J9_unknown", 0.0, 1000.0),
                _measurement("J9_unknown", 10.0, 1100.0),
                _measurement("J9_unknown", 20.0, 1200.0),
            ],
            {"joints": {"J1_base": {"theta_zero_deg": 0.0}}},
        )
