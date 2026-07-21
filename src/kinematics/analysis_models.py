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