from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JacobianResult:
    jacobian: np.ndarray
    scaled_jacobian: np.ndarray
    radial_distance_mm: float
    elbow_relative_angle_rad: float
    approach_angle_rad: float
    characteristic_length_mm: float

@dataclass(frozen=True)
class ManipulabilityMetrics:
    singular_values: np.ndarray
    rank: int
    smallest_singular_value: float
    largest_singular_value: float
    condition_number: float
    inverse_condition_number: float
    yoshikawa_manipulability: float
    normalized_manipulability: float
    weakest_task_direction: np.ndarray
    weakest_joint_direction: np.ndarray