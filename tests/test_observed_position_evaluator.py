from math import sqrt

import pytest

from kinematics.observed_position_evaluator import (
    PositionObservation,
    evaluate_observations,
    load_observations,
)


def test_reports_real_positioning_metrics() -> None:
    observations = [
        PositionObservation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), True),
        PositionObservation((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), True),
        PositionObservation((10.0, 0.0, 0.0), (10.0, 4.0, 0.0), True),
        PositionObservation((10.0, 0.0, 0.0), None, False),
    ]

    report = evaluate_observations(observations)

    assert report.total_attempts == 4
    assert report.successful_attempts == 3
    assert report.failed_attempts == 1
    assert report.success_rate == pytest.approx(0.75)
    assert report.axis_bias_x_mm == pytest.approx(4.0 / 3.0)
    assert report.axis_bias_y_mm == pytest.approx(4.0 / 3.0)
    assert report.axis_bias_z_mm == pytest.approx(0.0)
    assert report.mean_euclidean_error_mm == pytest.approx(8.0 / 3.0)
    assert report.rmse_mm == pytest.approx(sqrt(26.0 / 3.0))
    assert report.median_error_mm == pytest.approx(3.0)
    assert report.maximum_error_mm == pytest.approx(4.0)
    assert report.standard_deviation_mm == pytest.approx(1.2472191289)
    assert report.repeatability_mm == pytest.approx(1.0)
    assert report.repeatability_target_count == 1
    assert report.repeatability_sample_count == 2


def test_repeatability_requires_repeated_successful_target() -> None:
    report = evaluate_observations(
        [
            PositionObservation(
                (0.0, 0.0, 0.0),
                (1.0, 2.0, 3.0),
                True,
            )
        ]
    )

    assert report.repeatability_mm is None
    assert report.repeatability_target_count == 0
    assert report.repeatability_sample_count == 0


def test_all_failed_trials_still_report_success_rate() -> None:
    report = evaluate_observations(
        [PositionObservation((0.0, 0.0, 0.0), None, False)]
    )

    assert report.success_rate == 0.0
    assert report.mean_euclidean_error_mm is None
    assert report.axis_bias_x_mm is None


def test_loads_successes_and_failures_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "observations.csv"
    csv_path.write_text(
        "requested_x_mm,requested_y_mm,requested_z_mm,"
        "observed_x_mm,observed_y_mm,observed_z_mm,success\n"
        "100,200,30,101,198,31,true\n"
        "100,200,30,,,,false\n",
        encoding="utf-8",
    )

    observations = load_observations(csv_path)

    assert observations == [
        PositionObservation(
            (100.0, 200.0, 30.0),
            (101.0, 198.0, 31.0),
            True,
        ),
        PositionObservation((100.0, 200.0, 30.0), None, False),
    ]


def test_successful_csv_trial_requires_camera_position(tmp_path) -> None:
    csv_path = tmp_path / "observations.csv"
    csv_path.write_text(
        "requested_x_mm,requested_y_mm,requested_z_mm,"
        "observed_x_mm,observed_y_mm,observed_z_mm,success\n"
        "100,200,30,,,,true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires all observed coordinates"):
        load_observations(csv_path)


def test_rejects_non_finite_observation() -> None:
    with pytest.raises(ValueError, match="three finite coordinates"):
        PositionObservation(
            (0.0, 0.0, 0.0),
            (float("nan"), 0.0, 0.0),
            True,
        )
