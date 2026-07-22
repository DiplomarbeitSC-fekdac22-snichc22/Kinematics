from typing import Any


class AngleOutsideCalibrationError(ValueError):
    """Raised when an angle is outside the calibrated joint-angle range."""
    def __init__(
        self,
        angle_deg: float,
        minimum_deg: float,
        maximum_deg: float,
    ) -> None:
        self.angle_deg = angle_deg
        self.minimum_deg = minimum_deg
        self.maximum_deg = maximum_deg
        super().__init__(
            f"angle {angle_deg:g} deg outside calibrated range "
            f"{minimum_deg:g}-{maximum_deg:g} deg"
        )


class PulseLimitError(ValueError):
    """Raised when a calibrated angle requires an out-of-range pulse."""
    def __init__(
        self,
        pulse_us: int,
        minimum_us: int,
        maximum_us: int,
    ) -> None:
        self.pulse_us = pulse_us
        self.minimum_us = minimum_us
        self.maximum_us = maximum_us
        super().__init__(
            f"calibrated angle requires {pulse_us} us outside pulse range "
            f"{minimum_us}-{maximum_us} us"
        )


def clamp_angle(value: float, low: float, high: float) -> float:
    """Clamp a scalar for numerically bounded mathematical operations."""
    return max(low, min(high, value))


def angle_to_pwm_unclamped(angle_deg: float, joint: dict[str, Any]) -> float:
    """Return the raw calibrated pulse for diagnostics and analysis."""
    return joint["pulse_center_us"] + (
            (angle_deg - joint["theta_zero_deg"])
            * joint["direction"]
            * joint["us_per_degree"]
    )


def angle_to_pwm(angle_deg: float, joint: dict[str, Any]) -> int:
    """Convert a command angle, failing instead of silently clamping it."""
    minimum_deg = float(joint["theta_min_deg"])
    maximum_deg = float(joint["theta_max_deg"])
    angle_deg = float(angle_deg)

    if not minimum_deg <= angle_deg <= maximum_deg:
        raise AngleOutsideCalibrationError(
            angle_deg,
            minimum_deg,
            maximum_deg,
        )

    pulse_us = round(angle_to_pwm_unclamped(angle_deg, joint))
    minimum_us = int(joint["pulse_min_us"])
    maximum_us = int(joint["pulse_max_us"])

    if not minimum_us <= pulse_us <= maximum_us:
        raise PulseLimitError(pulse_us, minimum_us, maximum_us)

    return pulse_us


def angle_to_pwm_clamped(angle_deg: float, joint: dict[str, Any]) -> int:
    """Return an endpoint-clamped pulse for plots and visualizations only."""
    pulse = angle_to_pwm_unclamped(angle_deg, joint)
    return round(
        clamp_angle(
            pulse,
            joint["pulse_min_us"],
            joint["pulse_max_us"],
        )
    )
