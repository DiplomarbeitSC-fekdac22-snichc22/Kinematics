from typing import Any

def clamp_angle(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_to_pwm(angle_deg: float, joint: dict[str, Any]) -> int:
    pulse = joint["pulse_center_us"] + (
            (angle_deg - joint["theta_zero_deg"])
            * joint["direction"]
            * joint["us_per_degree"]
    )
    return round(clamp_angle(pulse, joint["pulse_min_us"], joint["pulse_max_us"]))