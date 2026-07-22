from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class PulseAngleMeasurement:
    """One physically measured joint angle at a commanded pulse width."""
    joint_name: str
    angle_deg: float
    pulse_us: float

    def __post_init__(self) -> None:
        if not self.joint_name.strip():
            raise ValueError("joint_name must not be empty")
        if not isfinite(self.angle_deg):
            raise ValueError("angle_deg must be finite")
        if not isfinite(self.pulse_us) or self.pulse_us <= 0.0:
            raise ValueError("pulse_us must be finite and positive")


@dataclass(frozen=True)
class JointCalibrationFit:
    """Linear pulse/angle fit and its physical-measurement diagnostics."""
    joint_name: str
    sample_count: int
    unique_angle_count: int
    theta_zero_deg: float
    slope_us_per_degree: float
    intercept_us: float
    direction: int
    us_per_degree: float
    pulse_center_us: float
    observed_angle_min_deg: float
    observed_angle_max_deg: float
    observed_pulse_min_us: float
    observed_pulse_max_us: float
    rmse_us: float
    mean_absolute_error_us: float
    maximum_absolute_error_us: float
    r_squared: float

    def predicted_pulse_us(self, angle_deg: float) -> float:
        """Predict a pulse from the fitted line without applying limits."""
        return self.intercept_us + self.slope_us_per_degree * angle_deg

    def as_dict(self) -> dict[str, Any]:
        """Return a stable representation for reports and review tooling."""
        return {
            "joint_name": self.joint_name,
            "suggested_updates": {
                "theta_zero_deg": self.theta_zero_deg,
                "direction": self.direction,
                "pulse_center_us": self.pulse_center_us,
                "us_per_degree": self.us_per_degree,
            },
            "fit": {
                "slope_us_per_degree": self.slope_us_per_degree,
                "intercept_us": self.intercept_us,
                "sample_count": self.sample_count,
                "unique_angle_count": self.unique_angle_count,
                "rmse_us": self.rmse_us,
                "mean_absolute_error_us": self.mean_absolute_error_us,
                "maximum_absolute_error_us": self.maximum_absolute_error_us,
                "r_squared": self.r_squared,
            },
            "observed_range": {
                "angle_min_deg": self.observed_angle_min_deg,
                "angle_max_deg": self.observed_angle_max_deg,
                "pulse_min_us": self.observed_pulse_min_us,
                "pulse_max_us": self.observed_pulse_max_us,
            },
        }
