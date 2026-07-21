import numpy as np

from kinematics.manipulability import calculate_manipulability_metrics


def test_condition_metrics_for_diagonal_matrix() -> None:
    matrix = np.diag([8.0, 4.0, 2.0, 1.0])
    result = calculate_manipulability_metrics(
        matrix,
        radial_distance_mm=100.0,
        elbow_relative_angle_rad=np.pi / 2.0,
        maximum_radial_reach_mm=200.0,
        rank_relative_tolerance=1e-12,
    )

    np.testing.assert_allclose(result.singular_values, [8.0, 4.0, 2.0, 1.0])
    assert result.rank == 4
    assert result.condition_number == 8.0
    assert result.inverse_condition_number == 0.125
    assert result.yoshikawa_manipulability == 64.0
    assert result.normalized_manipulability == 0.5


def test_singular_matrix_has_zero_inverse_condition() -> None:
    matrix = np.diag([8.0, 4.0, 2.0, 0.0])
    result = calculate_manipulability_metrics(
        matrix,
        radial_distance_mm=0.0,
        elbow_relative_angle_rad=0.0,
        maximum_radial_reach_mm=200.0,
        rank_relative_tolerance=1e-12,
    )

    assert result.rank == 3
    assert np.isinf(result.condition_number)
    assert result.inverse_condition_number == 0.0
    assert result.yoshikawa_manipulability == 0.0
