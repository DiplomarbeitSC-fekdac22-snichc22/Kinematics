from math import isfinite

import numpy as np

from kinematics.analysis_models import ManipulabilityMetrics


def calculate_manipulability_metrics(
        scaled_jacobian: np.ndarray,
        *,
        radial_distance_mm: float,
        elbow_relative_angle_rad: float,
        maximum_radial_reach_mm: float,
        rank_relative_tolerance: float
) -> ManipulabilityMetrics:
    """Calculate SVD, rank, condition, and manipulability metrics."""
    matrix = np.asarray(scaled_jacobian, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Scaled Jacobian must be 4x4, received {matrix.shape}")

    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)

    largest = float(singular_values[0])
    smallest = float(singular_values[-1])

    zero_threshold = rank_relative_tolerance * largest
    rank = int(np.count_nonzero(singular_values > zero_threshold))

    # kappa = sigma_max / sigma_min
    # It tends to infinity at a singularity
    condition_number = (
        float("inf")
        if smallest <= zero_threshold
        else largest / smallest
    )

    # mu = sigma_min / sigma_max is bounded between zero and one
    inverse_condition_number = (
        0.0
        if largest == 0.0 or not isfinite(condition_number)
        else smallest / largest
    )

    # Yoshikawa manipulability equals sqrt(det(J J^T))
    # For a square Jacobian = the product of all singular values
    yoshikawa = float(np.prod(singular_values))

    normalized = (
        abs(radial_distance_mm)
        / maximum_radial_reach_mm
        * abs(np.sin(elbow_relative_angle_rad))
    )

    weakest_task_direction = u[:, -1].copy()
    weakest_joint_direction = vh[-1, :].copy()

    return ManipulabilityMetrics(
        singular_values=singular_values,
        rank=rank,
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        condition_number=condition_number,
        inverse_condition_number=inverse_condition_number,
        yoshikawa_manipulability=yoshikawa,
        normalized_manipulability=float(normalized),
        weakest_task_direction=weakest_task_direction,
        weakest_joint_direction=weakest_joint_direction
    )
