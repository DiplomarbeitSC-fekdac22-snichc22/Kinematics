from math import radians

import numpy as np

from kinematics.forward_kinematics import calculate_gripper_center
from kinematics.jacobian import calculate_jacobian


REGULAR_ANGLES = {
    "J1_base": 15.0,
    "J2_shoulder": -100.0,
    "J3_elbow": 50.0,
    "J4_wrist": -30.0,
}


def _task_vector(joint_angles_deg: dict[str, float]) -> np.ndarray:
    position = calculate_gripper_center(joint_angles_deg)

    alpha_rad = radians(
        joint_angles_deg["J2_shoulder"]
        + 180.0
        - joint_angles_deg["J3_elbow"]
        + joint_angles_deg["J4_wrist"]
    )

    return np.array(
        [position["x_mm"], position["y_mm"], position["z_mm"], alpha_rad],
        dtype=float,
    )


def test_jacobian_has_expected_shape() -> None:
    result = calculate_jacobian(REGULAR_ANGLES)
    assert result.jacobian.shape == (4, 4)
    assert result.scaled_jacobian.shape == (4, 4)


def test_analytical_jacobian_matches_central_finite_difference() -> None:
    analytical = calculate_jacobian(REGULAR_ANGLES).jacobian

    epsilon_rad = 1e-6
    epsilon_deg = np.degrees(epsilon_rad)
    numerical = np.zeros((4, 4), dtype=float)
    joint_names = ["J1_base", "J2_shoulder", "J3_elbow", "J4_wrist"]

    for column, joint_name in enumerate(joint_names):
        plus = dict(REGULAR_ANGLES)
        minus = dict(REGULAR_ANGLES)
        plus[joint_name] += epsilon_deg
        minus[joint_name] -= epsilon_deg

        numerical[:, column] = (
            _task_vector(plus) - _task_vector(minus)
        ) / (2.0 * epsilon_rad)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-5)
