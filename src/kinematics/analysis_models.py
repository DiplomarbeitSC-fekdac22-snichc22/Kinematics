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

@dataclass(frozen=True)
class ConfigurationAnalysis:
    joint_angles_deg: dict[str, float]
    jacobian: JacobianResult
    metrics: ManipulabilityMetrics
    base_axis_singularity: bool
    elbow_singularity: bool
    geometric_status: str
    conditioning_status: str
    joint_limit_margin: float
    pulse_limit_margin: float
    constraint_status: str
    warnings: tuple[str, ...]